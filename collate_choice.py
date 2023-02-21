import argparse
import math
import random
import time

import torch
import numpy as np

from datasets import build_fag_dataset
# from datasets import build_gen_dataset
from util.argparser import get_args_parser


def main(args):
    hoi_len = set()

    dataset_train = build_fag_dataset(image_set="train", args=args)
    dataset_val = build_fag_dataset(image_set="val", args=args)
    for i in range(len(dataset_train)):
        _, anno_dict = dataset_train[i]
        hoi_len.add(len(anno_dict['valid_pairs']))
        # break
        assert len(anno_dict['valid_pairs']) == len(anno_dict['hoi_sentence'])
        if i > 100:
            break
        print(anno_dict['valid_pairs'])
    print(hoi_len)

    # print(round(100/7))
    # target = [(len(100) // 10) * 8:]
    # div = 12
    # split_len = 100 // div
    # ind = np.tile(np.arange(div), (split_len+1, 1)).T
    # print(ind.shape)
    # print(ind)
    # print(ind.ravel().shape)

    # start_time = time.time()
    #
    # div = 66
    # split_len = 100 // div
    # # split_len = math.ceil(100/div)
    # # print(split_len)
    # ind = [torch.rand(split_len) if i != div-1 else torch.rand(100-(div-1)*split_len) for i in range(div)]
    # ind_collate = torch.cat(ind)
    # print(ind_collate.size())
    #
    # total_time = time.time() - start_time
    # print(f'Training time {total_time}')
    # print(random.sample([1, 2, 3, 4], 3))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser('SentenceHOI training and evaluation script', parents=[get_args_parser()])
    arg = parser.parse_args()
    main(arg)
