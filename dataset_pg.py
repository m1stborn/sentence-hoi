import os
from collections import defaultdict
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets import build_dataset, build_gen_dataset
from engine import train_one_epoch, evaluate_hoi_fag
from models.hoitr import build as build_model
from util.argparser import get_args_parser
from datasets.hico_eval_triplet_from_json import HICOEvaluatorJson
from models.sentence_critreion import SentenceCriterion


train_target_fields = ['orig_size', 'size', 'boxes', 'labels', 'iscrowd', 'area', 'filename', 'hoi_sentence',
                       'hoi_candidate', 'obj_labels', 'verb_labels', 'hoi_labels', 'sub_boxes', 'obj_boxes']

val_target_fields = ['orig_size', 'size', 'anno', 'filename', 'boxes', 'labels', 'id', 'hois']


def main(args):
    # Fix the seed for reproducibility.
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # GEN-VL-KT dataset
    dataset_train = build_gen_dataset(image_set='train', args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # dataset_val = build_gen_dataset(image_set='val', args=args)
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # data_loader_val = DataLoader(dataset_val, 8, sampler=sampler_val,
    #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    # model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    assert args.backbone in ['resnet50', 'resnet101', 'swin'], args.backbone
    if args.backbone == 'resnet50':
        pretrain_model = './data/detr_coco/detr-r50-e632da11.pth'
    elif args.backbone == 'resnet101':
        pretrain_model = './data/detr_coco/detr-r101-2c7b67e5.pth'
    else:
        pretrain_model = None
    if pretrain_model is not None:
        pretrain_dict = torch.load(pretrain_model, map_location='cpu')['model']
        my_model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in my_model_dict}

        # convert pretrain weight
        pretrain_dict['query_embed.weight'] = pretrain_dict['query_embed.weight'].clone()[:args.num_queries]

        my_model_dict.update(pretrain_dict)
        model.load_state_dict(my_model_dict)

    # Optimizer
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            # TODO: require fix
            # optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    sen_criterion = SentenceCriterion(device=device)
    for i, (samples, targets) in enumerate(data_loader_train):
        samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in targets]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        outputs = model(samples)
        print(outputs.keys())
        # for t in targets:
        #     print(len(t['hoi_sentence']), len(t['hoi_candidate']), t['hoi_pair'], t['filename'])
        sen_loss = sen_criterion.batch_l1_loss(outputs, targets)
        print(sen_loss)
        pred = sen_criterion.inference(outputs)
        print(len(pred["collate_pred"][0]['hoi_prediction']))

        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SentenceHOI training and evaluation script', parents=[get_args_parser()])
    arg = parser.parse_args()
    main(arg)
