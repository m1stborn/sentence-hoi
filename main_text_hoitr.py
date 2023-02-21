import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import (
    get_scheduler,
)

import util.misc as utils
from datasets import build_gen_dataset, build_fag_dataset
from engine import train_one_epoch, evaluate_hoi_fag
from models.hoitr_text import build as build_model
from models.sentence_critreion import SentenceCriterion
from util.argparser import get_args_parser


def main(args):
    # Fix the seed for reproducibility.
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.use_fag_setting:
        # FG dataset
        dataset_train = build_fag_dataset(image_set="train", args=args)
        dataset_val = build_fag_dataset(image_set="val", args=args)
    else:
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

    # -----------------------------------------
    # dataset_fag_train = build_fag_dataset(image_set="train", args=args)
    # img_tensor, anno_dict = dataset_fag_train[0]
    # print("Fag dataset:")
    # for k, v in anno_dict.items():
    #     if torch.is_tensor(v):
    #         print(k, ":", v.size())
    #     else:
    #         print(k, ":", v)
    # print("Gen dataset:")
    # img_tensor, anno_dict = dataset_train[0]
    # for k, v in anno_dict.items():
    #     if torch.is_tensor(v):
    #         print(k, ":", v.size())
    #     else:
    #         print(k, ":", v)
    # -----------------------------------------

    # Model
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    # Dev model
    # output = model(torch.rand((1, 3, 512, 560)).to(args.device))
    # targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in [anno_dict]]
    # losses = criterion(output, targets)
    # return

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
        my_model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in my_model_dict}

        # convert pretrain weight
        pretrain_dict['query_embed.weight'] = pretrain_dict['query_embed.weight'].clone()[:args.num_queries]

        my_model_dict.update(pretrain_dict)
        model.load_state_dict(my_model_dict)

    # Optimizer
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.5)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.epochs * len(data_loader_train),
    )

    print(f"Num training steps: {args.epochs * len(data_loader_train)}")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint \
                and 'epoch' in checkpoint and not args.with_sentence_branch:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        args.start_epoch = checkpoint['epoch'] + 1
        print(f"Load model from Epoch {checkpoint['epoch']}")

    accelerator = Accelerator()

    # Prepare everything with our `accelerator`.
    model, optimizer, data_loader_train, data_loader_val, lr_scheduler, criterion, sen_criterion = accelerator.prepare(
        model, optimizer, data_loader_train, data_loader_val, lr_scheduler, criterion, sen_criterion
    )

    # Train
    print("Start training")
    start_time = time.time()
    best_performance = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device,
            args, epoch, lr_scheduler, sen_criterion, args.clip_max_norm)
        # lr_scheduler.step()

        if epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint_last.pth')
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        if epoch < 100 and epoch % 2 != 0:  # eval every 5 epoch before lr_drop
            continue

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
