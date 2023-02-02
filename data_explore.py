import argparse
import random
from collections import defaultdict
from pathlib import Path
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from datasets import build_gen_dataset
from engine_gen import evaluate_hoi
from models.hoitr import build as build_model
from models.sentence_critreion import SentenceCriterion
from util.argparser import get_args_parser
from datasets.hico_text_label import hico_text_label, hico_obj_text_label


def main(args):
    # Fix the seed for reproducibility.
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pair2text = hico_text_label
    text_list = list(pair2text.values())
    text2idx = {text: text_list.index(text) for text in pair2text.values()}

    obj_id2text = {k: v for (k, v) in hico_obj_text_label}

    hoi_label = []
    hoi_triplets = []
    hoi_sentences = []
    hoi_verbs = []
    hoi_obj_ids = []
    hoi_obj_texts = []
    hoi_counts = []
    hoi_train_counts = []
    hoi_eval_counts = []
    hoi_in_train = []
    # GEN-VL-KT dataset
    dataset_train = build_gen_dataset(image_set='train', args=args)
    dataset_val = build_gen_dataset(image_set='val', args=args)
    print(len(dataset_val))
    for i, (img, target) in enumerate(dataset_train):
        print(i)
        # hoi_sentence = target['hoi_sentence']
        for pair in target['hoi_pair']:
            pair = (pair[0], int(pair[1]))
            # print(pair, hoi_sentence)
            # print(hico_text_label.get(pair, None))
            sentence = hico_text_label.get(pair, "A photo of unseen interaction")
            if pair not in hoi_triplets:
                hoi_label.append(text2idx.get(sentence, -1))
                hoi_triplets.append(pair)
                hoi_sentences.append(sentence)
                hoi_verbs.append(pair[0])
                hoi_obj_ids.append(pair[1])
                hoi_counts.append(1)
                hoi_train_counts.append(1)
                hoi_eval_counts.append(0)
                hoi_obj_texts.append(obj_id2text[pair[1]])
                hoi_in_train.append(True)
            else:
                index = hoi_triplets.index(pair)
                hoi_counts[index] += 1
                hoi_train_counts[index] += 1
    eval_pair_count = 0
    for i, (img, target) in enumerate(dataset_val):
        hois = target['hois'].numpy()
        eval_pair_count += len(hois)
        for pair in hois:
            pair = (pair[0], pair[1])
            sentence = hico_text_label.get(pair, "A photo of unseen interaction.")
            if pair not in hoi_triplets:
                hoi_label.append(text2idx.get(sentence, -1))
                hoi_triplets.append(pair)
                hoi_sentences.append(sentence)
                hoi_verbs.append(pair[0])
                hoi_obj_ids.append(pair[1])
                hoi_counts.append(1)
                hoi_train_counts.append(0)
                hoi_eval_counts.append(1)
                hoi_obj_texts.append(obj_id2text.get(pair[1], "A photo of unseen object."))
                hoi_in_train.append(False)
            else:
                index = hoi_triplets.index(pair)
                hoi_counts[index] += 1
                hoi_eval_counts[index] += 1
    print(eval_pair_count)
    data = {
        "label": hoi_label,
        "triplet": hoi_triplets,
        "sentence": hoi_sentences,
        "verb_id": hoi_verbs,
        "object_id": hoi_obj_ids,
        "object_text": hoi_obj_texts,
        "count": hoi_counts,
        "train_count": hoi_train_counts,
        "eval_count": hoi_eval_counts,
        "in_train": hoi_in_train,
    }
    df = pd.DataFrame.from_dict(data)
    df.to_csv("./checkpoint/hoi.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SentenceHOI training and evaluation script', parents=[get_args_parser()])
    arg = parser.parse_args()
    main(arg)
