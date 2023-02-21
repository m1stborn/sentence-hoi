import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from datasets import build_gen_dataset
from datasets import build_fag_dataset
from engine import evaluate_hoi_fag
from models.hoitr import build as build_model
from models.sentence_critreion import SentenceCriterion
from util.argparser import get_args_parser


def main(args):
    checkpoint = torch.load(args.resume, map_location='cpu')
    # overwrite args
    args = checkpoint['args']
    print(f"Load model from Epoch {checkpoint['epoch']}")

    # Fix the seed for reproducibility.
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # For loading old version checkpoint p_202301261744
    # Since model layer change form hoi_visual_projection.weight to verb_cls_embed.weight
    if "hoi_visual_projection.weight" in checkpoint['model']:
        args.use_fag_setting = False
        args.with_sentence_branch = False

    if args.use_fag_setting:
        # FG dataset
        dataset_val = build_fag_dataset(image_set="val", args=args)
    else:
        # GEN-VL-KT dataset
        dataset_val = build_gen_dataset(image_set='val', args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, 8, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    model, _, postprocessors = build_model(args)
    model.to(device)
    if "hoi_visual_projection.weight" in checkpoint['model']:
        print(f"Try to load old version of hoi_visual_projection, convert parameter.")
        my_model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in checkpoint['model'].items() if k in my_model_dict}
        pretrain_dict['verb_cls_embed.weight'] = checkpoint['model']['hoi_visual_projection.weight'].clone()
        pretrain_dict['verb_cls_embed.bias'] = checkpoint['model']['hoi_visual_projection.bias'].clone()
        model.load_state_dict(pretrain_dict, strict=False)
    else:
        model.load_state_dict(checkpoint['model'])

    print(f"Load model from Epoch {checkpoint['epoch']}")

    test_stats = evaluate_hoi_fag(args.dataset_file, model, postprocessors, data_loader_val,
                                  args.subject_category_id, device, args)
    print(test_stats)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SentenceHOI training and evaluation script', parents=[get_args_parser()])
    arg = parser.parse_args()
    if arg.output_dir:
        arg.output_dir = arg.output_dir+"_test"
        Path(arg.output_dir).mkdir(parents=True, exist_ok=True)
    main(arg)
