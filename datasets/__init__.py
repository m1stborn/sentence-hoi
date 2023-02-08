# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .hico import build as build_hico
# from .hoia import build as build_hoia
# from .vcoco import build as build_vcoco
from .hico_gen import build as build_hico_gen
from .hico_fag import build as build_hico_fag


def build_dataset(image_set, args, test_scale=-1):
    assert args.dataset_file in ['hico', 'vcoco', 'hoia'], args.dataset_file
    if args.dataset_file == 'hico':
        return build_hico(image_set, test_scale)
    # elif args.dataset_file == 'vcoco':
    #     return build_vcoco(image_set, test_scale)
    # else:
    #     return build_hoia(image_set, test_scale)


def build_gen_dataset(image_set, args):
    if args.dataset_file == 'hico':
        return build_hico_gen(image_set, args)
    # if args.dataset_file == 'vcoco':
    #     return build_vcoco(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')


def build_fag_dataset(image_set, args):
    if args.dataset_file == 'hico':
        return build_hico_fag(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
