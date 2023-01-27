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


def main(args):
    # Fix the seed for reproducibility.
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # GEN-VL-KT dataset
    dataset_train = build_gen_dataset(image_set='train', args=args)
    # dataset_val = build_gen_dataset(image_set='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
    #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    for i, batch in enumerate(dataset_train):
        img, annotation = batch
        print(annotation['size'], img.size(), annotation['orig_size'])
        if i > 10:
            break

    # for i, batch in enumerate(data_loader_train):
    #     img, annotation = batch
    #     print(img.tensor)
    #     if i > 10:
    #         break


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SentenceHOI training and evaluation script', parents=[get_args_parser()])
    arg = parser.parse_args()
    if arg.output_dir:
        Path(arg.output_dir).mkdir(parents=True, exist_ok=True)
    main(arg)
