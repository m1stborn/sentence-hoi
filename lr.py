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
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    get_scheduler,
)

import util.misc as utils
from datasets import build_dataset, build_gen_dataset, build_fag_dataset
from engine import train_one_epoch, evaluate_hoi_fag
from models.hoitr import build as build_model
from models.sentence_critreion import SentenceCriterion
# from models.synth_sentence_criterion import SentenceCriterion
from util.argparser import get_args_parser


def main():
    # device = torch.device(args.device)
    #
    # model, criterion, postprocessors = build_model(args)
    # model.to(device)
    #
    # optimizer = torch.optim.AdamW(model.parameters(), lr=7.5e-5)
    # # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.5)
    #
    # lr_scheduler = get_scheduler(
    #     name="cosine",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=150 * 2352,
    # )
    # data = {"epoch": [],
    #         "lr": []}
    #
    # for e in range(150):
    #     data["epoch"].append(e)
    #     data["lr"].append(lr_scheduler.get_lr())
    #     for s in range(2352):
    #         lr_scheduler.step()
    #
    # df = pd.DataFrame.from_dict(data)
    # df.to_csv("./data/cosine_lr75.csv", index=False)
    for epoch in range(0, 10):
        if epoch < 100 and epoch % 2 != 0 and epoch != 10-1:
            continue
        print(epoch, "eval")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser('SentenceHOI training and evaluation script', parents=[get_args_parser()])
    # arg = parser.parse_args()
    # if arg.output_dir:
    #     Path(arg.output_dir).mkdir(parents=True, exist_ok=True)
    # main(arg)
    main()
