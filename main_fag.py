import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
# from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler
# from transformers import (
#     get_scheduler,
# )

import util.misc as utils
from datasets import build_gen_dataset, build_fag_dataset
from engine import train_one_epoch, evaluate_hoi_fag
from models.hoitr import build as build_model
from models.sentence_critreion import SentenceCriterion
from util.argparser import get_args_parser, save_args

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
GEN dataset structure:
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
        # hoi_candidate ['a photo of a person racing a motorcycle', 
        #             'a photo of a person riding a motorcycle', 
        #             'a photo of a person sitting on a motorcycle', 
        #             'a photo of a person straddling a motorcycle']
    }
"""

"""
Fag dataset structure:
    (img_tensor, annot_dict)
:param img_tensor:
    torch.Tensor, size: torch.Size([3, 512, 512])
:param anno_dict: dict {
        orig_size : torch.Size([2]) same
        size : torch.Size([2]) same
        boxes : torch.Size([2, 4]) same
        labels : torch.Size([2]) same
        iscrowd : torch.Size([2]) same
        area : torch.Size([2]) same
        obj_labels : torch.Size([1]) same
        verb_labels : torch.Size([1, 117]) same
        sub_boxes : torch.Size([1, 4]) same
        obj_boxes : torch.Size([1, 4]) same
        
        verb_label_enc : torch.Size([117])
        
        hoi_pairs: List[(verb, obj)] # can be map by obj2id.json and verb2id.json
        
        # compare to gen dataset
        orig_size : torch.Size([2])
        size : torch.Size([2])
        boxes : torch.Size([2, 4])
        labels : torch.Size([2])
        iscrowd : torch.Size([2])
        area : torch.Size([2])
        obj_labels : torch.Size([1])
        verb_labels : torch.Size([1, 117])
        sub_boxes : torch.Size([1, 4])
        obj_boxes : torch.Size([1, 4])
        
        filename : HICO_train2015_00000001.jpg 
        hoi_sentence : ['a photo of a person racing a motorcycle']
        hoi_labels : torch.Size([1, 600])
    }
"""


def main(args):
    utils.init_distributed_mode(args)

    # Fix the seed for reproducibility.
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    save_args(args)

    if args.use_fag_setting:
        # FG dataset
        dataset_train = build_fag_dataset(image_set="train", args=args)
        dataset_val = build_fag_dataset(image_set="val", args=args)
    else:
        # GEN-VL-KT dataset
        dataset_train = build_gen_dataset(image_set='train', args=args)
        dataset_val = build_gen_dataset(image_set='val', args=args)

    if args.distributed:
        print("using DistributedSampler")
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 16, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # Model
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    sen_criterion = None
    if args.with_sentence_branch:
        sen_criterion = SentenceCriterion(device=device)

    assert args.backbone in ['resnet50', 'resnet101', 'swin'], args.backbone
    if args.backbone == 'resnet50':
        pretrain_model = './data/detr_coco/detr-r50-e632da11.pth'
    elif args.backbone == 'resnet101':
        pretrain_model = './data/detr_coco/detr-r101-2c7b67e5.pth'
    else:
        pretrain_model = None
    if pretrain_model is not None:
        pretrain_dict = torch.load(pretrain_model, map_location='cpu')['model']
        my_model_dict = model_without_ddp.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in my_model_dict}

        # convert pretrain weight
        pretrain_dict['query_embed.weight'] = pretrain_dict['query_embed.weight'].clone()[:args.num_queries]

        my_model_dict.update(pretrain_dict)
        model_without_ddp.load_state_dict(my_model_dict)

    # Optimizer
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.1)
    # lr_scheduler = get_scheduler(
    #     name="cosine",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=args.epochs * len(data_loader_train),
    # )

    print(f"Num training steps: {args.epochs * len(data_loader_train)}")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        print(f"Load model from Epoch {checkpoint['epoch']}")
        if "hoi_visual_projection.weight" in checkpoint['model']:
            print(f"Try to load old version of hoi_visual_projection, covert parameter.")
            my_model_dict = model.state_dict()
            pretrain_dict = {k: v for k, v in checkpoint['model'].items() if k in my_model_dict}
            pretrain_dict['verb_cls_embed.weight'] = checkpoint['model']['hoi_visual_projection.weight'].clone()
            pretrain_dict['verb_cls_embed.bias'] = checkpoint['model']['hoi_visual_projection.bias'].clone()
            model_without_ddp.load_state_dict(pretrain_dict, strict=False)
        else:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint \
                    and 'epoch' in checkpoint and not args.with_sentence_branch:
                # optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        args.start_epoch = checkpoint['epoch'] + 1

    # accelerator = Accelerator()

    # Prepare everything with our `accelerator`.
    # model, optimizer, data_loader_train, data_loader_val, lr_scheduler, criterion, sen_criterion = accelerator.prepare(
    #     model, optimizer, data_loader_train, data_loader_val, lr_scheduler, criterion, sen_criterion
    # )

    # Train
    print("Start training")
    start_time = time.time()
    best_performance = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device,
            args, epoch, lr_scheduler, sen_criterion, args.clip_max_norm)
        lr_scheduler.step()

        if epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint_last.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        # if epoch < 4 and  1 < epoch:
        #     if epoch % 2 != 0:
        #         continue
        if epoch < 10 or epoch >= 60:
            if epoch % 2 != 0:
                continue
        elif epoch < 60:
            if epoch % 5 != 0 and epoch != args.epochs-1:  # eval every 5 epoch before lr_drop
                continue
        # elif epoch % 2 != 0:
        #     continue

        test_stats = evaluate_hoi_fag(args.dataset_file, model, postprocessors, data_loader_val,
                                      args.subject_category_id, device, args)

        performance = None
        if args.dataset_file == 'hico':
            performance = test_stats['mAP_def']
        # elif args.dataset_file == 'vcoco':
        #     performance = test_stats['mAP_all']
        # elif args.dataset_file == 'hoia':
        #     performance = test_stats['mAP']

        if performance > best_performance:
            print(f"Saving epoch {epoch}!")
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
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
