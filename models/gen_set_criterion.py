import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (accuracy, get_world_size)


def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _neg_loss(pred, gt, weights=None, alpha=0.25):
    """ Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little more memory
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    loss = 0

    pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    if weights is not None:
        pos_loss = pos_loss * weights[:-1]

    neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.clip_model, _ = clip.load(args.clip_model, device=device)
        self.alpha = args.alpha

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = _get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    # @torch.no_grad()
    # def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
    #     pred_logits = outputs['pred_obj_logits']
    #     device = pred_logits.device
    #     tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
    #     card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
    #     card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
    #     losses = {'obj_cardinality_error': card_err}
    #     return losses

    # def loss_verb_labels(self, outputs, targets, indices, num_interactions):
    #     assert 'pred_verb_logits' in outputs
    #     src_logits = outputs['pred_verb_logits']
    #
    #     idx = _get_src_permutation_idx(indices)
    #     target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
    #     target_classes = torch.zeros_like(src_logits)
    #     target_classes[idx] = target_classes_o
    #
    #     src_logits = src_logits.sigmoid()
    #     loss_verb_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
    #     losses = {'loss_verb_ce': loss_verb_ce}
    #     return losses

    def loss_hoi_labels(self, outputs, targets, indices, num_interactions, topk=5):
        assert 'pred_hoi_logits' in outputs
        src_logits = outputs['pred_hoi_logits']

        idx = _get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['hoi_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o
        src_logits = _sigmoid(src_logits)
        loss_hoi_ce = _neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_hoi_labels': loss_hoi_ce}

        _, pred = src_logits[idx].topk(topk, 1, True, True)
        acc = 0.0
        for tid, target in enumerate(target_classes_o):
            tgt_idx = torch.where(target == 1)[0]
            if len(tgt_idx) == 0:
                continue
            acc_pred = 0.0
            for tgt_rel in tgt_idx:
                acc_pred += (tgt_rel in pred[tid])
            acc += acc_pred / len(tgt_idx)
        rel_labels_error = 100 - 100 * acc / max(len(target_classes_o), 1)
        losses['hoi_class_error'] = torch.from_numpy(np.array(
            rel_labels_error)).to(src_logits.device).float()

        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = _get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                    exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)

        return losses

    # def mimic_loss(self, outputs, targets, indices, num_interactions):
    #     src_feats = outputs['inter_memory']
    #     src_feats = torch.mean(src_feats, dim=1)
    #
    #     target_clip_inputs = torch.cat([t['clip_inputs'].unsqueeze(0) for t in targets])
    #     with torch.no_grad():
    #         target_clip_feats = self.clip_model.encode_image(target_clip_inputs)
    #     loss_feat_mimic = F.l1_loss(src_feats, target_clip_feats)
    #     losses = {'loss_feat_mimic': loss_feat_mimic}
    #     return losses

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'hoi_labels': self.loss_hoi_labels,
            'obj_labels': self.loss_obj_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes,
        }
        # if 'pred_hoi_logits' in outputs.keys():
        #     loss_map = {
        #         'hoi_labels': self.loss_hoi_labels,
        #         'obj_labels': self.loss_obj_labels,
        #         'sub_obj_boxes': self.loss_sub_obj_boxes,
        #         # 'feats_mimic': self.mimic_loss
        #     }
        # else:
        #     loss_map = {
        #         'obj_labels': self.loss_obj_labels,
        #         'obj_cardinality': self.loss_obj_cardinality,
        #         'verb_labels': self.loss_verb_labels,
        #         'sub_obj_boxes': self.loss_sub_obj_boxes,
        #     }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        """
        :return:
            losses: dict {
                'loss_hoi_labels', 'hoi_class_error', 'loss_obj_ce', 'obj_class_error', 'loss_sub_bbox',
                'loss_obj_bbox', 'loss_sub_giou', 'loss_obj_giou', 'loss_hoi_labels_0', 'hoi_class_error_0',
                'loss_obj_ce_0', 'loss_sub_bbox_0', 'loss_obj_bbox_0', 'loss_sub_giou_0', 'loss_obj_giou_0',
                'loss_hoi_labels_1', 'hoi_class_error_1', 'loss_obj_ce_1', 'loss_sub_bbox_1', 'loss_obj_bbox_1',
                'loss_sub_giou_1', 'loss_obj_giou_1']
            }
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['hoi_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            result = self.get_loss(loss, outputs, targets, indices, num_interactions)
            # print(f"loss:{loss}, {result}")
            losses.update(result)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessHOITriplet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_hoi_logits = outputs['pred_hoi_logits']
        out_obj_logits = outputs['pred_obj_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']

        assert len(out_hoi_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        hoi_scores = out_hoi_logits.sigmoid()
        obj_scores = out_obj_logits.sigmoid()
        obj_labels = F.softmax(out_obj_logits, -1)[..., :-1].max(-1)[1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(hoi_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(hoi_scores)):
            hs, os, ol, sb, ob = hoi_scores[index], obj_scores[index], obj_labels[index], sub_boxes[index], obj_boxes[
                index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            ids = torch.arange(b.shape[0])

            results[-1].update({'hoi_scores': hs.to('cpu'), 'obj_scores': os.to('cpu'),
                                'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:]})

        return results
