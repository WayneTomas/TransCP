from typing import Dict, Tuple
import torch
import numpy as np

import sys
import torch.nn.functional as F

from util.box_utils import bbox_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou, ciou_loss


def build_target(args, gt_bbox, pred, device):
    batch_size = gt_bbox.size(0)
    num_scales = len(pred)
    coord_list, bbox_list = [], []
    for scale_ii in range(num_scales):
        this_stride = 32 // (2**scale_ii)
        grid = args.size // this_stride
        # Convert [x1, y1, x2, y2] to [x_c, y_c, w, h]
        center_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2
        center_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2
        box_w = gt_bbox[:, 2] - gt_bbox[:, 0]
        box_h = gt_bbox[:, 3] - gt_bbox[:, 1]
        coord = torch.stack((center_x, center_y, box_w, box_h), dim=1)
        # Normalized by the image size
        coord = coord / args.size
        coord = coord * grid
        coord_list.append(coord)
        bbox_list.append(torch.zeros(coord.size(0), 3, 5, grid, grid))

    best_n_list, best_gi, best_gj = [], [], []
    for ii in range(batch_size):
        anch_ious = []
        for scale_ii in range(num_scales):
            this_stride = 32 // (2**scale_ii)
            grid = args.size // this_stride
            # gi = coord_list[scale_ii][ii,0].long()
            # gj = coord_list[scale_ii][ii,1].long()
            # tx = coord_list[scale_ii][ii,0] - gi.float()
            # ty = coord_list[scale_ii][ii,1] - gj.float()
            gw = coord_list[scale_ii][ii, 2]
            gh = coord_list[scale_ii][ii, 3]

            anchor_idxs = [x + 3 * scale_ii for x in [0, 1, 2]]
            anchors = [args.anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                x[1] / (args.anchor_imsize/grid)) for x in anchors]

            ## Get shape of gt box
            # gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # import pdb
            # pdb.set_trace()

            gt_box = torch.from_numpy(np.array([0, 0, gw.cpu().numpy(), gh.cpu().numpy()])).float().unsqueeze(0)
            ## Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))

            ## Calculate iou between gt and anchor shapes
            anch_ious += list(bbox_iou(gt_box, anchor_shapes))
        ## Find the best matching anchor box
        best_n = np.argmax(np.array(anch_ious))
        best_scale = best_n // 3

        best_grid = args.size // (32 / (2**best_scale))
        anchor_idxs = [x + 3 * best_scale for x in [0, 1, 2]]
        anchors = [args.anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [ (x[0] / (args.anchor_imsize/best_grid), \
            x[1] / (args.anchor_imsize/best_grid)) for x in anchors]

        gi = coord_list[best_scale][ii, 0].long()
        gj = coord_list[best_scale][ii, 1].long()
        tx = coord_list[best_scale][ii, 0] - gi.float()
        ty = coord_list[best_scale][ii, 1] - gj.float()
        gw = coord_list[best_scale][ii, 2]
        gh = coord_list[best_scale][ii, 3]
        tw = torch.log(gw / scaled_anchors[best_n % 3][0] + 1e-16)
        th = torch.log(gh / scaled_anchors[best_n % 3][1] + 1e-16)

        bbox_list[best_scale][ii, best_n % 3, :, gj, gi] = torch.stack([tx, ty, tw, th, torch.ones(1).to(device).squeeze()])
        best_n_list.append(int(best_n))
        best_gi.append(gi)
        best_gj.append(gj)

    for ii in range(len(bbox_list)):
        bbox_list[ii] = bbox_list[ii].to(device)
    return bbox_list, best_gi, best_gj, best_n_list


def yolo_loss(pred_list, target, gi, gj, best_n_list, device, w_coord=5., w_neg=1. / 5, size_average=True):
    mseloss = torch.nn.MSELoss(size_average=True)
    celoss = torch.nn.CrossEntropyLoss(size_average=True)
    num_scale = len(pred_list)
    batch_size = pred_list[0].size(0)

    pred_bbox = torch.zeros(batch_size, 4).to(device)
    gt_bbox = torch.zeros(batch_size, 4).to(device)
    for ii in range(batch_size):
        pred_bbox[ii, 0:2] = torch.sigmoid(pred_list[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, 0:2, gj[ii], gi[ii]])
        pred_bbox[ii, 2:4] = pred_list[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, 2:4, gj[ii], gi[ii]]
        gt_bbox[ii, :] = target[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, :4, gj[ii], gi[ii]]
    loss_x = mseloss(pred_bbox[:, 0], gt_bbox[:, 0])
    loss_y = mseloss(pred_bbox[:, 1], gt_bbox[:, 1])
    loss_w = mseloss(pred_bbox[:, 2], gt_bbox[:, 2])
    loss_h = mseloss(pred_bbox[:, 3], gt_bbox[:, 3])

    pred_conf_list, gt_conf_list = [], []
    for scale_ii in range(num_scale):
        pred_conf_list.append(pred_list[scale_ii][:, :, 4, :, :].contiguous().view(batch_size, -1))
        gt_conf_list.append(target[scale_ii][:, :, 4, :, :].contiguous().view(batch_size, -1))
    pred_conf = torch.cat(pred_conf_list, dim=1)
    gt_conf = torch.cat(gt_conf_list, dim=1)
    loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
    return (loss_x + loss_y + loss_w + loss_h) * w_coord + loss_conf


def trans_vg_loss(batch_pred, batch_target):
    """Compute the losses related to the bounding boxes, 
       including the L1 regression loss and the GIoU loss
    """
    batch_size = batch_pred.shape[0]
    # world_size = get_world_size()
    num_boxes = batch_size

    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')
    # loss_bbox = F.smooth_l1_loss(batch_pred, batch_target, reduction='none')
    loss_giou = 1 - torch.diag(generalized_box_iou(xywh2xyxy(batch_pred), xywh2xyxy(batch_target)))
    # 注意这里由于引用的其他人实现的ciou，出现了loss函数行为的不一致，
    # loss_ciou = torch.diag(ciou_loss(xywh2xyxy(batch_pred), xywh2xyxy(batch_target)))

    losses = {}
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes
    losses['loss_giou'] = loss_giou.sum() / num_boxes
    # losses['loss_ciou'] = loss_ciou.sum() / num_boxes

    return losses


def prototype_loss(x: torch.Tensor, y: torch.Tensor, scale_output=True):
    """ 
    """
    import math
    # cos_sim = torch.cosine_similarity(x, y, dim=1)
    loss = F.mse_loss(x, y)
    # loss = torch.mean(F.mse_loss(x, y))
    # hsic = RbfHSIC(sigma_x=1, sigma_y=1)
    # loss = hsic.unbiased_estimator(x, y)
    return {"loss_dis": loss}


def classify_loss(seg_logits: torch.Tensor, target_label: torch.Tensor) -> Dict:
    # h, w = fea_size
    # scaled_target = F.interpolate(batch_target[None].float(), size=(h, w))[0].view(-1, h * w)
    # loss = F.binary_cross_entropy_with_logits(batch_pred_conx, scaled_target) + F.binary_cross_entropy_with_logits(
    #     batch_pred_sub, scaled_target)
    # loss = F.binary_cross_entropy_with_logits(seg_logits.squeeze(), target_label)
    loss = F.binary_cross_entropy(seg_logits.squeeze(), target_label.float())
    return {"seg_loss": loss}


def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def crossview_contrastive_loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    #     Works with pytorch <= 1.2
    #     p_i_j[(p_i_j < EPS).data] = EPS
    #     p_j[(p_j < EPS).data] = EPS
    #     p_i[(p_i < EPS).data] = EPS

    # Works with pytorch > 1.2
    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))

    loss = 0.1 * loss.sum()

    return {"crossview_contrastive_loss": loss}

def infoNCE_loss(h_i, h_j, temperature_f):
    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask


    h_i = F.normalize(h_i, dim=1)
    h_j = F.normalize(h_j, dim=1)
    N = 2 * h_i.size(0)
    h = torch.cat((h_i, h_j), dim=0)

    sim = torch.matmul(h, h.T) / temperature_f
    sim_i_j = torch.diag(sim, h_i.size(0))
    sim_j_i = torch.diag(sim, -h_i.size(0))

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(N)
    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduce='sum')
    loss /= N

    return {"infoNCE_loss": loss}
