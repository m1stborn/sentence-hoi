import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from datasets import build_gen_dataset, build_fag_dataset
from util.argparser import get_args_parser, save_args


def main(args):
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

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 8, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # img, anno = dataset_train[0]

    print(dataset_train.annotations[0])

    # print(dataset_train.bg_image_filename)
    # bg_idx = dataset_train.bg_image_filename
    # bg_filename = []
    # for i in bg_idx:
    #     anno = dataset_train.annotations[dataset_train.ids[i]]
    #     bg_filename.append(anno['file_name'])

    # with open("../gen-vlkt/data/hico_20160224_det/annotations/bg_image_idx.json", "w", encoding="utf-8") as file:
    #     json.dump({
    #         "bg_image_idx": bg_idx,
    #         "bg_image_filename": bg_filename,
    #     }, file, ensure_ascii=False, indent=4)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SentenceHOI training and evaluation script', parents=[get_args_parser()])
    arg = parser.parse_args()
    if arg.output_dir:
        Path(arg.output_dir).mkdir(parents=True, exist_ok=True)
    main(arg)
