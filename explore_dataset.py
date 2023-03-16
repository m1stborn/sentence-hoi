import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    # for i, idx in enumerate(range(len(dataset_train)-1, 0, -1)):
    #     img, anno = dataset_train[idx]
    #     # print(anno["file_name"])
    #     print(anno['filename'])
    #     print(idx)
    #     if i > 100:
    #         break
    print(dataset_train.annotations[0])

    hoi_list = json.load(open(os.path.join("./data/annotations/", 'hoi_list_new.json'), 'r'))
    with open("./data/annotations/hoi_id_to_num.json", "r") as file:
        hoi_rare_mapping = json.load(file)
    rare_img_list = []  # boolean list
    for i, anno in tqdm(enumerate(dataset_train.annotations)):
        is_rare = False
        hoi_label = []
        for hoi in anno['hoi_annotation']:
            hoi_id = hoi['hoi_category_id'] - 1
            str_id = hoi_list[hoi_id]['id']
            hoi_dict = hoi_rare_mapping[str_id]
            hoi_label.append({
                "id": str_id,
                **hoi_dict,
            })
            if hoi_dict['rare'] == True:
                is_rare = True
                break
        rare_img_list.append(is_rare)
        print(anno['file_name'], hoi_label, is_rare)
        # if i > 100:
        #     break

    # with open("./data/annotations/train_img_contains_rare_hoi.json", "w", encoding="utf-8") as file:
    #     json.dump(rare_img_list, file, ensure_ascii=False, indent=4)
    # print(np.sum(rare_img_list))
    # assert len(rare_img_list) == len(dataset_train.annotations)

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
