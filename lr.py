import os
from collections import defaultdict
import argparse
import datetime
import json
import random
import time
from pathlib import Path

from datasets import build_gen_dataset
from util.argparser import get_args_parser


# import numpy as np
# import pandas as pd
# from accelerate import Accelerator
# import torch
# from torch.utils.data import DataLoader, DistributedSampler
# from transformers import (
#     get_scheduler,
# )

# import util.misc as utils
# from datasets import build_dataset, build_gen_dataset, build_fag_dataset
# from engine import train_one_epoch, evaluate_hoi_fag
# from models.hoitr import build as build_model
# from models.sentence_critreion import SentenceCriterion
# # from models.synth_sentence_criterion import SentenceCriterion
# from util.argparser import get_args_parser


def main(args):
    dataset_train = build_gen_dataset(image_set='train', args=args)
    dataset_val = build_gen_dataset(image_set='val', args=args)
    print(dataset_train[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SentenceHOI training and evaluation script', parents=[get_args_parser()])
    arg = parser.parse_args()
    if arg.output_dir:
        Path(arg.output_dir).mkdir(parents=True, exist_ok=True)
    main(arg)
