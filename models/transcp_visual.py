import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.parameter import Parameter

import math

from .neck import PrototypeLearner
from .v_transformer import build_v_transformer

class BboxRegression(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        args = cfg.copy()
        layer_type = args.pop('type')
        self.layer = _MODULES[layer_type](**args)

        self.norm = nn.LayerNorm(256)
        self.num_visu_token = 400
        num_total = self.num_visu_token + 1
        self.v_transformer = build_v_transformer(args)
        self.v_pos_embed = nn.Embedding(num_total, 256)
        self.reg_token = nn.Embedding(1, 256)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_feat, img_key_padding_mask, pos_embed, word_feat, word_mask, projected_disentangled_lang, h, w):
        hw, bs, c = img_feat.shape
        # Visual Context Disentangling + Prototype Learning
        x_multi_modal = self.layer(img_feat, img_key_padding_mask, pos_embed,
                          word_feat, word_mask, projected_disentangled_lang, h, w)
        
        # Box reggresion (without box prediction head)
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)

        v_src = torch.cat([tgt_src, x_multi_modal], dim=0)
        v_mask = torch.cat([tgt_mask, img_key_padding_mask], dim=1)
        v_pos = self.v_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.v_transformer(v_src, v_mask, v_pos)  # (1+L+N)xBxC
        if self.norm is not None:
            hs = self.norm(output)
        return hs[0]


class VisualDenstanglingPrototype(nn.Module):
    def __init__(self, num_queries, query_dim,
                 return_intermediate=False,
                 extra_layer=None, num_extra_layers=1):
        super().__init__()

        args = extra_layer.copy()
        layer_type = args.pop('type')
        extra_encoder_layer = _MODULES[layer_type](**args)
        self.extra_encoder_layers = _get_clones(
            extra_encoder_layer, num_extra_layers)

        self.return_intermediate = return_intermediate
        self.vis_query_embed = nn.Embedding(num_queries, query_dim)
        self.text_query_embed = nn.Embedding(num_queries, query_dim)

        # prototype discovery module
        self.prototypelearner = PrototypeLearner(num_tokens=2048, decay=0.4)
        

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, img_feat, img_key_padding_mask=None, pos=None,
                word_feat=None, word_key_padding_mask=None, projected_disentangled_lang=None, h=20, w=20):
        hw, bs, c = img_feat.shape

        # Visual Context Disentangling (Encode discriminative features)
        for layer in self.extra_encoder_layers:
            img_feat = layer(img_feat, img_key_padding_mask, pos,
                             word_feat, word_key_padding_mask, None)
        
        img_feat_srcs = img_feat.chunk(2, dim=-1)
        dis_img = img_feat_srcs[1]
        ori_img = img_feat_srcs[0]


        # prototype embedding
        prototype_out = self.prototypelearner(dis_img, h, w)
        dis_img_vd = prototype_out["embedded_pt"]

        x_multi_modal = torch.tanh(dis_img_vd.permute(1, 2, 0).contiguous().view(bs, -1, w, h)) * torch.tanh(projected_disentangled_lang)
        x_multi_modal = x_multi_modal.view(bs, -1, hw).permute(2, 0, 1)
        return x_multi_modal


class DiscriminativeFeatEncLayer(nn.Module):
    def __init__(self, d_model, img2text_attn_args=None, img_query_with_pos=True,
                discrimination_coef_settings=None):
        super().__init__()
        args = img2text_attn_args.copy()
        self.img2text_attn = MULTIHEAD_ATTNS[args.pop('type')](**args)
        self.img_query_with_pos = img_query_with_pos

        self.text_proj = MLP(**discrimination_coef_settings['text_proj'])
        self.img_proj = MLP(**discrimination_coef_settings['img_proj'])
        self.tf_pow = discrimination_coef_settings.get('pow')
        self.tf_scale = Parameter(torch.Tensor([discrimination_coef_settings.get('scale')]))
        self.tf_sigma = Parameter(torch.Tensor([discrimination_coef_settings.get('sigma')]))
        self.norm_img = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, img_feat, img_key_padding_mask, img_pos,
                word_feat, word_key_padding_mask, word_pos=None):
        orig_img_feat = img_feat

        # discrimination coeficient calculation
        img_query = img_feat + img_pos if self.img_query_with_pos else img_feat

        # shared semantic between vision and language, Eq. (3)
        F_s = self.img2text_attn(
            query=img_query, key=self.with_pos_embed(word_feat, word_pos),
            value=word_feat, key_padding_mask=word_key_padding_mask)[0]

        # it can be seen as the salient objects in the language features 
        # (e.g. "white dog besides the cat", "dog" is the salient object word in the language)
        text_embed = self.text_proj(F_s)

        # 
        img_embed = self.img_proj(img_feat)

        # Eq. (4)
        dis_coef = (F.normalize(img_embed, p=2, dim=-1) *
                        F.normalize(text_embed, p=2, dim=-1)).sum(dim=-1, keepdim=True)
        dis_coef = self.tf_scale * \
            torch.exp(- (1 - dis_coef).pow(self.tf_pow)
                      / (2 * self.tf_sigma**2))

        # Eq. (5)
        fuse_img_feat = self.norm_img(img_feat) * dis_coef
        return torch.cat([orig_img_feat, fuse_img_feat], dim=-1)


_MODULES = {
    'VisualDenstanglingPrototype': VisualDenstanglingPrototype,
    'DiscriminativeFeatEncLayer': DiscriminativeFeatEncLayer,
}


def build_bbox_regression(args):
    return BboxRegression(args.model_config['decoder'])


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu_inplace":
#         return nn.ReLU(inplace=True)
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MHAttentionRPE(nn.Module):
    ''' With relative position embedding '''

    def __init__(self, d_model, h, dropout=0.1, return_raw_attention=False,
                 pos_x_range=[-20, 20], pos_y_range=[-20, 20], pos_index_offset=20,
                 learnable_pos_embed=False):
        super().__init__()
        self.d_k = d_model // h
        self.h = h
        self.scaling = float(self.d_k) ** -0.5
        self.return_raw_attention = return_raw_attention

        self.in_proj_weight = Parameter(torch.Tensor(3 * d_model, d_model))
        self.in_proj_bias = Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn = None
        # self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout
        self._reset_parameters()

        self.learnable_pos_embed = learnable_pos_embed
        if learnable_pos_embed:
            self.pos_x = nn.Embedding(
                pos_x_range[1] - pos_x_range[0] + 1, d_model // 2)
            self.pos_y = nn.Embedding(
                pos_y_range[1] - pos_y_range[0] + 1, d_model // 2)
        else:
            pos_x, pos_y = position_embedding_sine(d_model // 2, normalize=True,
                                                   x_range=pos_x_range, y_range=pos_y_range)
            self.register_buffer('pos_x', pos_x)  # [x_range, C]
            self.register_buffer('pos_y', pos_y)  # [y_range, C]

        self.pos_index_offset = pos_index_offset

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None):
        tgt_len, bs, dim = query.size()
        src_len, _, dim = key.size()

        weight_q, bias_q = self.in_proj_weight[0:dim], self.in_proj_bias[0:dim]
        weight_k, bias_k = self.in_proj_weight[dim:dim *
                                               2], self.in_proj_bias[dim:dim*2]
        weight_v, bias_v = self.in_proj_weight[dim *
                                               2:], self.in_proj_bias[dim*2:]

        q = query.matmul(weight_q.t()) + bias_q
        k = key.matmul(weight_k.t()) + bias_k
        v = value.matmul(weight_v.t()) + bias_v

        # [bs*h, tgt_len, dim//h]
        q = q.view(tgt_len, bs * self.h, -1).transpose(0, 1)
        # [bs*h, dim//h, src_len], To calculate qTk (bmm)
        k = k.view(src_len, bs * self.h, -1).permute(1, 2, 0)
        v = v.view(src_len, bs * self.h, -1).transpose(0, 1)

        q = q * self.scaling
        attn_weights = torch.bmm(q, k)  # [bs*h, tgt_len, src_len]

        # compute the relative positions
        bs, HW = key_padding_mask.size()
        assert (HW == 400) and (HW == tgt_len)
        img_mask = ~key_padding_mask.view(bs, 20, 20)
        yy = img_mask.cumsum(1, dtype=torch.float32).view(
            bs, -1)  # [bs, HW],  1~20
        xx = img_mask.cumsum(2, dtype=torch.float32).view(
            bs, -1)  # [bs, HW],  1~20
        diff_yy = yy[:, :, None] - yy[:, None, :]  # [bs, HW, HW]
        diff_xx = xx[:, :, None] - xx[:, None, :]  # [bs, HW, HW]
        if self.learnable_pos_embed:
            k_posy = self.pos_y.weight.matmul(
                weight_k.t()[:dim//2])  # [x_range, dim]
            k_posx = self.pos_x.weight.matmul(
                weight_k.t()[dim//2:])  # [y_range, dim]
        else:
            k_posy = self.pos_y.matmul(weight_k.t()[:dim//2])  # [x_range, dim]
            k_posx = self.pos_x.matmul(weight_k.t()[dim//2:])  # [y_range, dim]
        k_posy = k_posy.view(-1, 1, self.h, dim//self.h).repeat(1, bs, 1, 1).\
            reshape(-1, bs * self.h, dim//self.h).permute(1,
                                                          2, 0)  # [bs*h, dim//h, y_range]
        k_posx = k_posx.view(-1, 1, self.h, dim//self.h).repeat(1, bs, 1, 1).\
            reshape(-1, bs * self.h, dim//self.h).permute(1,
                                                          2, 0)  # [bs*h, dim//h, x_range]
        posy_attn_weights = torch.bmm(q, k_posy).view(
            bs, self.h, HW, -1)  # [bs, h, HW, y_range]
        posx_attn_weights = torch.bmm(q, k_posx).view(
            bs, self.h, HW, -1)  # [bs, h, HW, x_range]
        diff_yy_idx = diff_yy[:, None].repeat(
            1, self.h, 1, 1) + self.pos_index_offset
        diff_xx_idx = diff_xx[:, None].repeat(
            1, self.h, 1, 1) + self.pos_index_offset

        posy_attn_weights = torch.gather(
            posy_attn_weights, -1, diff_yy_idx.long())  # [bs, h, HW, HW]
        posx_attn_weights = torch.gather(
            posx_attn_weights, -1, diff_xx_idx.long())  # [bs, h, HW, HW]
        pos_attn_weights = (posy_attn_weights +
                            posx_attn_weights).view(bs*self.h, HW, -1)
        attn_weights = attn_weights + pos_attn_weights

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(-1, self.h, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(
                    2),  # [bs, 1, 1, src_len]
                float('-inf')
            )
            attn_weights = attn_weights.view(-1, tgt_len, src_len)
        raw_attn_weights = attn_weights
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout_p, training=self.training)
        attn_output = torch.bmm(attn_weights, v)
        self.attn = attn_weights

        attn_output = attn_output.transpose(
            0, 1).contiguous().view(tgt_len, bs, -1)
        attn_output = F.linear(
            attn_output, self.out_proj.weight, self.out_proj.bias)
        if self.return_raw_attention:
            return attn_output, raw_attn_weights
        return attn_output, attn_weights


MULTIHEAD_ATTNS = {
    'MultiheadAttention': nn.MultiheadAttention,
    'MHAttentionRPE': MHAttentionRPE,
}


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        if num_layers > 0:
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip(
                [input_dim] + h, h + [output_dim]))
        else:
            self.layers = []

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def position_embedding_sine(num_pos_feats=64, temperature=10000, normalize=False, scale=None,
                            x_range=[-20, 20], y_range=[-20, 20], device=None):
    if scale is not None and normalize is False:
        raise ValueError("normalize should be True if scale is passed")
    if scale is None:
        scale = 2 * math.pi

    x_embed = torch.arange(x_range[0], x_range[1] + 1, device=device)
    y_embed = torch.arange(y_range[0], y_range[1] + 1, device=device)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[-1] + eps) * scale
        x_embed = x_embed / (x_embed[-1] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, None] / dim_t
    pos_y = y_embed[:, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=-1).flatten(1)
    pos_y = torch.stack(
        (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=-1).flatten(1)
    return pos_x, pos_y
