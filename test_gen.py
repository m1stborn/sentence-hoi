import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from datasets import build_gen_dataset
from engine_gen import evaluate_hoi
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
    dataset_val = build_gen_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # TODO: load model
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(f"Load model from Epoch {checkpoint['epoch']}")

    test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val,
                              args.subject_category_id, device, args)
    print(test_stats)

    # Evaluate
    # evaluator = HICOEvaluatorJson("./checkpoint/p_202301160140/results.json", dataset_val.rare_triplets,
    #                               dataset_val.non_rare_triplets, args=args)

    # evaluator.evaluate()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SentenceHOI training and evaluation script', parents=[get_args_parser()])
    arg = parser.parse_args()
    if arg.output_dir:
        arg.output_dir = arg.output_dir+"_test"
        Path(arg.output_dir).mkdir(parents=True, exist_ok=True)
    main(arg)
