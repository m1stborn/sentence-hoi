import torch
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_xyxy_to_cxcywh
from util.misc import (accuracy, get_world_size, is_dist_avail_and_initialized)
from .segmentation import (sigmoid_focal_loss)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def dis(x):
    return x[-1]


def sample(num, points):
    if num == 0:
        result = []
    else:
        while len(points) < num:
            points.append(([0, 0], 0))
        space = int(len(points) / num)
        points.sort(key=dis)
        m = 0
        sample_points = []
        for i in range(num):
            sample_points.append(np.array(points[m + i * space][0]))
        result = np.array(sample_points)
    return result


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses,
                 focal_alpha=0.25,
                 no_obj=False):
        super().__init__()
        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.no_obj = no_obj
        self.distance = torch.nn.PairwiseDistance(p=2)
        if self.no_obj:
            empty_weight = torch.ones(self.num_obj_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)

    def loss_hoi_score_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_hoi_score' in outputs
        pred_hoi_score = outputs['pred_hoi_score']
        idx = _get_src_permutation_idx(indices)
        target_hoi_score = torch.zeros_like(pred_hoi_score)
        target_hoi_score[idx] = torch.tensor([1], dtype=torch.float32, device=pred_hoi_score.device)
        loss_hoi_score_ce = sigmoid_focal_loss(pred_hoi_score, target_hoi_score, num_interactions,
                                               alpha=self.focal_alpha, gamma=2) * pred_hoi_score.shape[1]
        losses = {'loss_hoi_score_ce': loss_hoi_score_ce}
        return losses

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = _get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        if not self.no_obj:
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout,
                                                device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_obj_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_interactions,
                                             alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        else:
            loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']
        idx = _get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()

        loss_verb_ce = _neg_loss(src_logits, target_classes)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_anchor(self, outputs, targets, indices, num_interactions):
        assert 'dynamic_anchor' in outputs
        idx = _get_src_permutation_idx(indices)
        src_dynamic_anchor = outputs['dynamic_anchor'][idx]
        target_sub_points = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_points = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_interaction_points = (target_obj_points[:, :2] + target_sub_points[:, :2]) / 2
        distance = self.distance(src_dynamic_anchor, target_interaction_points)
        loss = distance.mean()
        losses = {'loss_anchor': loss}
        return losses

    def hoi_box(self, sub_boxes, obj_boxes):
        # sub_boxes  x y w h
        # obj_boxes  x y w h
        # return hoi_boxes x y w h
        h = box_cxcywh_to_xyxy(sub_boxes)
        o = box_cxcywh_to_xyxy(obj_boxes)
        hoi_boxes = torch.tensor(
            [[min(h[i][0], o[i][0]), min(h[i][1], o[i][1]), max(h[i][2], o[i][2]), max(h[i][3], o[i][3])] for i in
             range(sub_boxes.shape[0])],
            dtype=torch.float32, device=sub_boxes.device)  # x y x y
        if hoi_boxes.shape[0] == 0:
            return torch.tensor([], dtype=torch.float32, device=sub_boxes.device)
        return box_xyxy_to_cxcywh(hoi_boxes)  # x y w h

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = _get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]  # x y w h
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        # src_hoi_boxes = outputs['pred_hoi_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # target_hoi_boxes = self.hoi_box(target_sub_boxes, target_obj_boxes)
        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)
        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            # losses['loss_hoi_bbox'] = src_hoi_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
            # losses['loss_hoi_giou'] = src_hoi_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            # loss_hoi_bbox = F.l1_loss(src_hoi_boxes, target_hoi_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                        exist_obj_boxes.sum() + 1e-4)
            # losses['loss_hoi_bbox'] = loss_hoi_bbox.sum() / num_interactions
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            # loss_hoi_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_hoi_boxes),
            #                                                    box_cxcywh_to_xyxy(target_hoi_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
            # losses['loss_hoi_giou'] = loss_hoi_giou.sum() / num_interactions
        return losses

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if not loss == "verb_label_enc":
                losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "verb_label_enc":
                        continue
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            if "verb_label_enc" in self.losses:
                l_dict = self.get_loss("verb_label_enc", enc_outputs, targets, None, None)
                l_dict = {k: v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
