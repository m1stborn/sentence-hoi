# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
DETR model and criterion classes.
"""
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .transformer import build_transformer

from .gen_matcher import build_matcher as build_gen_matcher
from .gen_set_criterion import SetCriterionHOI as GenSetCriterionHOI
from .gen_set_criterion import PostProcessHOITriplet, PostProcessHOIFag

num_humans = 2


class HoiTR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries,
                 aux_loss=False, sentence_branch=False, use_fag_setting=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_obj_classes: number of object classes
            num_verb_classes: number of verb classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            sentence_branch: hoi, sentence, both.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.human_box_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.object_box_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.object_cls_embed = nn.Linear(hidden_dim, num_obj_classes) if use_fag_setting \
            else nn.Linear(hidden_dim, num_obj_classes + 1)

        self.use_fag_setting = use_fag_setting
        print(f"Interaction action head setting: {'fag' if self.use_fag_setting else 'gen'}")
        self.verb_cls_embed = nn.Linear(hidden_dim, num_verb_classes if self.use_fag_setting else 600)
        # Deprecated layer, the old checkpoint has it. Can be removed after a fresh train.
        # self.human_cls_embed = nn.Linear(hidden_dim, num_humans + 1)
        # self.action_cls_embed = nn.Linear(hidden_dim, num_actions + 1)
        # self.hoi_visual_projection = nn.Linear(hidden_dim, 600)

        self.sentence_branch = sentence_branch
        if sentence_branch:
            self.sentence_embed = MLP(hidden_dim, hidden_dim, 512, 4)

        self.hoi_concept = 'pred_verb_logits' if self.use_fag_setting else 'pred_hoi_logits'

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        # hs: hidden state = torch.Size([num_dec_layer, batch_size, num_query, hidden_dim])=torch.Size([3, 1, 100, 256])

        human_outputs_coord = self.human_box_embed(hs).sigmoid()
        object_outputs_class = self.object_cls_embed(hs)
        object_outputs_coord = self.object_box_embed(hs).sigmoid()
        verb_outputs_class = self.verb_cls_embed(hs)

        # Deprecated layer, the old checkpoint has it. Can be removed after a fresh train.
        # human_outputs_class = self.human_cls_embed(hs)
        # hoi_outputs_class = self.hoi_visual_projection(hs)

        out = {
            'pred_sub_boxes': human_outputs_coord[-1],
            'pred_obj_logits': object_outputs_class[-1],
            'pred_obj_boxes': object_outputs_coord[-1],
            # 'pred_hoi_logits': hoi_outputs_class[-1],
            self.hoi_concept: verb_outputs_class[-1],
        }
        # if self.use_fag_setting:
        #     verb_outputs_class = self.verb_cls_embed(hs)
        #     # torch.Size([dec_layer, batch_size, num_query, num_verb_class+1])
        #     out['pred_verb_logits'] = verb_outputs_class[-1]
        # else:
        #     verb_outputs_class = self.hoi_visual_projection(hs)
        #     out['pred_hoi_logits'] = verb_outputs_class[-1]

        if self.sentence_branch:
            sentence_embedding = self.sentence_embed(hs)
            out['pred_hoi_embeddings'] = sentence_embedding[-1]

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                human_outputs_coord,
                object_outputs_class,
                object_outputs_coord,
                # hoi_outputs_class,
                verb_outputs_class,
            )
        return out

    @torch.jit.unused
    def _set_aux_loss(self,
                      human_outputs_coord,
                      object_outputs_class,
                      object_outputs_coord,
                      # hoi_outputs_class,
                      verb_outputs_class,
                      ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # hoi_concept = 'pred_verb_logits' if self.use_fag_setting else 'pred_hoi_logits'
        return [{
            'pred_sub_boxes': a,
            'pred_obj_logits': b,
            'pred_obj_boxes': c,
            # 'pred_hoi_logits': d,
            # 'pred_verb_logits': d,
            self.hoi_concept: d,
        } for a, b, c, d in
            zip(
                human_outputs_coord[:-1],
                object_outputs_class[:-1],
                object_outputs_coord[:-1],
                # hoi_outputs_class[:-1],
                verb_outputs_class[:-1]
            )]


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


def build(args) -> Tuple[HoiTR, GenSetCriterionHOI, Union[PostProcessHOITriplet, PostProcessHOIFag]]:
    assert args.dataset_file in ['hico', 'vcoco', 'hoia'], args.dataset_file
    if args.dataset_file in ['hico']:
        # num_classes = 91   # less
        # num_actions = 118  # same with gen
        num_classes = 80
        num_actions = 117
        num_hoi = 600

    elif args.dataset_file in ['vcoco']:
        num_classes = 91
        num_actions = 30
    else:
        num_classes = 12
        num_actions = 11

    device = torch.device(args.device)

    if args.backbone == 'swin':
        from .backbone_swin import build_backbone_swin
        backbone = build_backbone_swin(args)
    else:
        backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = HoiTR(
        backbone,
        transformer,
        num_obj_classes=num_classes,
        num_verb_classes=num_actions,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        sentence_branch=args.with_sentence_branch,
        use_fag_setting=args.use_fag_setting
    )

    # GenSetCriterion
    matcher = build_gen_matcher(args)

    weight_dict = {}
    if args.with_clip_label:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef
    else:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef

    weight_dict['loss_verb_ce'] = args.hoi_loss_coef
    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef

    if args.with_sentence_branch:
        weight_dict['loss_sentence_l1'] = args.sentence_l1_loss_coef
        # weight_dict['loss_triplet'] = args.sentence_triplet_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # TODO: remove
    # losses = ['hoi_labels', 'obj_labels', 'sub_obj_boxes', 'verb_labels', 'obj_cardinality']
    if args.use_fag_setting:
        losses = ['obj_labels', 'sub_obj_boxes', 'verb_labels', 'obj_cardinality']
    else:
        losses = ['hoi_labels', 'obj_labels', 'sub_obj_boxes', 'obj_cardinality']

    criterion = GenSetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                   weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                   args=args)
    criterion.to(device)
    # postprocessors = {'hoi': PostProcessHOITriplet(args)}
    if args.use_fag_setting:
        postprocessors = PostProcessHOIFag(subject_category_id=args.subject_category_id, no_obj=False)
        # default subject_category_id = 0, no_obj = False
    else:
        postprocessors = PostProcessHOITriplet(args)
    return model, criterion, postprocessors
