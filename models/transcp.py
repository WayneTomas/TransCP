import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, get_world_size, is_dist_avail_and_initialized)
from pytorch_pretrained_bert.modeling import BertModel

from .backbone import build_backbone
from .transformer import build_visual_encoder
from .transcp_visual import build_bbox_regression
from .transcp_text import build_LSTMBert

class TransCP(nn.Module):
    def __init__(self, pretrained_weights, args=None):
        """ Initializes the model."""
        super().__init__()

        # Image feature encoder (CNN + Transformer encoder)
        self.backbone = build_backbone(args)
        self.trans_encoder = build_visual_encoder(args)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.trans_encoder.d_model, kernel_size=1)

        # language feature encoder (12 layer BERT) + lingustic context disentangling (Bi-GRU+PhraseAttention)
        self.bert = BertModel.from_pretrained(args.bert_model)
        self.bert_proj = nn.Linear(args.bert_output_dim, args.hidden_dim)
        self.bert_output_layers = args.bert_output_layers
        # we combine language feature encoder and language context disentangling within self.textmodel to make it concise
        self.textmodel = build_LSTMBert(args, bert_model=self.bert)
        # 

        self.bbox_regression = build_bbox_regression(args)

        hidden_dim = self.trans_encoder.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if pretrained_weights:
            self.load_pretrained_weights(pretrained_weights)
        
        # #####################################################
        # Add by 唐伟
        self.num_visu_token = 400
        self.lstm_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

    def load_pretrained_weights(self, weights_path):
        def load_weights(module, prefix, weights):
            module_keys = module.state_dict().keys()
            weights_keys = [k for k in weights.keys() if prefix in k]
            update_weights = dict()
            for k in module_keys:
                prefix_k = prefix+'.'+k
                if prefix_k in weights_keys:
                    update_weights[k] = weights[prefix_k]
                else:
                    print(f"Weights of {k} are not pre-loaded.")
            module.load_state_dict(update_weights, strict=False)

        weights = torch.load(weights_path, map_location='cpu')['model']
        load_weights(self.backbone, prefix='backbone', weights=weights)
        load_weights(self.trans_encoder, prefix='transformer', weights=weights)
        load_weights(self.input_proj, prefix='input_proj', weights=weights)
        print(f"Weights of DETR are pre-loaded.")


    def forward(self, image, image_mask, word_id, word_mask):
        # 1. Image features
        # Eq. (1)
        features, pos = self.backbone(NestedTensor(image, image_mask))
        src, mask = features[-1].decompose()
        bs, c, h, w = src.size()
        assert mask is not None
        img_feat, mask, pos_embed = self.trans_encoder(self.input_proj(src), mask, pos[-1])

        # 2. Text features (including disentangled language features)
        # In the implementation, we concat text encoder and lingustic context disentangling together
        # i.e. language feature encoder (12 layer BERT) + lingustic context disentangling (Bi-GRU+PhraseAttention)
        # Eq. (2) and Eq. (6) ~ Eq. (8)
        disentangled_lang, bert_fea = self.textmodel(NestedTensor(word_id, word_mask))
        word_feat, word_mask = bert_fea.decompose()
        word_feat = self.bert_proj(word_feat)
        word_feat = word_feat.permute(1, 0, 2) # NxLxC -> LxNxC
        assert word_mask is not None
        word_mask = word_mask.flatten(1)
        projected_disentangled_lang = self.lstm_proj(disentangled_lang).squeeze().unsqueeze(-1).unsqueeze(-1)
        
        # 3. visual context disentangling + prototype discovery + bbox regression (without box prediction head)
        # Eq. (3) ~ Eq. (14) 
        hs = self.bbox_regression(img_feat, mask, pos_embed, word_feat, word_mask, projected_disentangled_lang, h, w)
        
        # 4. box prediction head
        # Eq. (15)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_boxes': outputs_coord.unsqueeze(1)}
        return out




class VGCriterion(nn.Module):
    """ This class computes the loss for VLTVG."""
    def __init__(self, weight_dict, loss_loc, box_xyxy):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()
        self.weight_dict = weight_dict

        self.box_xyxy = box_xyxy

        self.loss_map = {'loss_boxes': self.loss_boxes}

        self.loss_loc = self.loss_map[loss_loc]

    def loss_boxes(self, outputs, target_boxes, num_pos):
        """Compute the losses related to the bounding boxes (the L1 regression loss and the GIoU loss)"""
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes'] # [B, #query, 4]
        target_boxes = target_boxes[:, None].expand_as(src_boxes)

        src_boxes = src_boxes.reshape(-1, 4) # [B*#query, 4]
        target_boxes = target_boxes.reshape(-1, 4) #[B*#query, 4]

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['l1'] = loss_bbox.sum() / num_pos

        if not self.box_xyxy:
            src_boxes = box_ops.box_cxcywh_to_xyxy(src_boxes)
            target_boxes = box_ops.box_cxcywh_to_xyxy(target_boxes)
        loss_giou = 1 - box_ops.box_pair_giou(src_boxes, target_boxes)
        losses['giou'] = (loss_giou[:, None]).sum() / num_pos
        return losses


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        """
        gt_boxes = targets['bbox']
        pred_boxes = outputs['pred_boxes']

        losses = {}
        B, Q, _ = pred_boxes.shape
        num_pos = avg_across_gpus(pred_boxes.new_tensor(B*Q))
        loss = self.loss_loc(outputs, gt_boxes, num_pos)
        losses.update(loss)

        # Apply the loss function to the outputs from all the stages
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                l_dict = self.loss_loc(aux_outputs, gt_boxes, num_pos)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format we expect"""
    def __init__(self, box_xyxy=False):
        super().__init__()
        self.bbox_xyxy = box_xyxy

    @torch.no_grad()
    def forward(self, outputs, target_dict):
        """ Perform the computation"""
        rsz_sizes, ratios, orig_sizes = \
            target_dict['size'], target_dict['ratio'], target_dict['orig_size']
        dxdy = None if 'dxdy' not in target_dict else target_dict['dxdy']

        boxes = outputs['pred_boxes']

        assert len(boxes) == len(rsz_sizes)
        assert rsz_sizes.shape[1] == 2

        boxes = boxes.squeeze(1)

        # Convert to absolute coordinates in the original image
        if not self.bbox_xyxy:
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        img_h, img_w = rsz_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct
        if dxdy is not None:
            boxes = boxes - torch.cat([dxdy, dxdy], dim=1)
        boxes = boxes.clamp(min=0)
        ratio_h, ratio_w = ratios.unbind(1)
        boxes = boxes / torch.stack([ratio_w, ratio_h, ratio_w, ratio_h], dim=1)
        if orig_sizes is not None:
            orig_h, orig_w = orig_sizes.unbind(1)
            boxes = torch.min(boxes, torch.stack([orig_w, orig_h, orig_w, orig_h], dim=1))

        return boxes


def avg_across_gpus(v, min=1):
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(v)
    return torch.clamp(v.float() / get_world_size(), min=min).item()


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_vgmodel(args):
    device = torch.device(args.device)

    model = TransCP(pretrained_weights=args.load_weights_path, args=args)

    weight_dict = {'loss_cls': 1, 'l1': args.bbox_loss_coef}
    weight_dict['giou'] = args.giou_loss_coef
    weight_dict.update(args.other_loss_coefs)
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = VGCriterion(weight_dict=weight_dict, loss_loc=args.loss_loc, box_xyxy=args.box_xyxy)
    criterion.to(device)

    postprocessor = PostProcess(args.box_xyxy)

    return model, criterion, postprocessor
