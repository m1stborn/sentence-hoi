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
from engine_gen import train_one_epoch, evaluate_hoi
from models.hoitr import build as build_model
from util.argparser import get_args_parser
# from engine import train_one_epoch, evaluate_hoi
# from models import build_model

# from datasets.hico_eval_triplet_from_json import HICOEvaluatorJson


"""
model output:
    human_pred_logits: torch.Size([3, 100, 3]) = [batch_size, num_queries, num_classes+1]
    
    object_pred_logits: torch.Size([3, 100, 92])
    human_pred_boxes: torch.Size([3, 100, 4])
    object_pred_boxes: torch.Size([3, 100, 4])
    action_pred_logits: torch.Size([3, 100, 119])
    
gen_model output
    pred_hoi_logits : torch.Size([3, 64, 600]) = [batch_size, num_queries, classes]
    pred_obj_logits : torch.Size([3, 64, 81]) -> object_pred_logits
    pred_sub_boxes : torch.Size([3, 64, 4]) -> human_pred_boxes
    pred_obj_boxes : torch.Size([3, 64, 4]) -> object_pred_boxes
"""


"""
dataset structure:
    (img_tensor, annot_dict)
:param img_tensor:
    torch.Tensor, size: torch.Size([3, 512, 512])
:param anno_dict:
    dict: {
        orig_size : torch.Size([2])
        filename: str
        size : torch.Size([2])
        boxes : torch.Size([2, 4])
        labels : torch.Size([2])
        iscrowd : torch.Size([2])
        area : torch.Size([2])
        sub_boxes : torch.Size([1, 4]) -> human_boxes 
        obj_boxes : torch.Size([1, 4]) -> object_boxes
        obj_labels : torch.Size([1])
        verb_labels : torch.Size([1, 117]) -> onehot encoding
        hoi_labels : torch.Size([1, 600]) -> onehot encoding
        hoi_sentence ['a photo of a person racing a motorcycle']
        hoi_candidate ['a photo of a person racing a motorcycle', 
                    'a photo of a person riding a motorcycle', 
                    'a photo of a person sitting on a motorcycle', 
                    'a photo of a person straddling a motorcycle']
        # clip_inputs : torch.Size([3, 224, 224])
        
        # "human_boxes": torch.Size([2, 4]), e.g. torch.Size([num_human, 4])
        # 'human_labels': torch.Size([2]), e.g. torch.Size([num_human])
        # 'object_boxes':torch.Size([2, 4]), e.g. torch.Size([num_obj, 4])
        # 'object_labels': torch.Size([2]), e.g. torch.Size([num_obj])
        # 'action_boxes':torch.Size([2, 4]), e.g. torch.Size([num_action, 4])
        # 'action_labels': torch.Size([2]), e.g. torch.Size([num_action])
    }
"""


def main(args):
    # Fix the seed for reproducibility.
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # GEN-VL-KT dataset
    dataset_train = build_gen_dataset(image_set='train', args=args)
    dataset_val = build_gen_dataset(image_set='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 8, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # print(dataset_val.rare_triplets, dataset_val.non_rare_triplets)
    # img_tensor, anno_dict = dataset_train[0]
    # for k, v in anno_dict.items():
    #     if torch.is_tensor(v):
    #         print(k, v.size())
    #     else:
    #         print(k, v)

    # hoiT dataset
    # dataset_train = build_dataset(image_set='train', args=args)
    # img_tensor, anno_dict = dataset_train[0]
    # for k, v in anno_dict.items():
    #     if torch.is_tensor(v):
    #         print(k, v)
    #     else:
    #         print(k, v)

    # Model
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
    # assert args.num_queries == 100, args.num_queries
    # assert args.enc_layers == 6 and args.dec_layers == 6
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
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            # TODO: require fix
            # optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # Dev model
    # output = model(torch.rand((1, 3, 512, 560)).to(args.device))
    # targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in [anno_dict]]
    # losses = criterion(output, targets)

    # Train
    print("Start training")
    start_time = time.time()
    best_performance = 0
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, args, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint_last.pth')
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        if epoch < args.lr_drop and epoch % 2 != 0:  # eval every 5 epoch before lr_drop
            continue
        # elif epoch >= args.lr_drop and epoch % 2 == 0:  # eval every 2 epoch after lr_drop
        #     continue

        test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val,
                                  args.subject_category_id, device, args)
        performance = None
        if args.dataset_file == 'hico':
            performance = test_stats['mAP']
        # elif args.dataset_file == 'vcoco':
        #     performance = test_stats['mAP_all']
        # elif args.dataset_file == 'hoia':
        #     performance = test_stats['mAP']

        if performance > best_performance:
            print(f"Saving epoch {epoch}!")
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

            best_performance = performance

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with open(f"{args.output_dir}/log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.dev and epoch == args.start_epoch:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SentenceHOI training and evaluation script', parents=[get_args_parser()])
    arg = parser.parse_args()
    if arg.output_dir:
        Path(arg.output_dir).mkdir(parents=True, exist_ok=True)
    main(arg)
