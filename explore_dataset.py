import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import util.misc as utils
from datasets import build_gen_dataset, build_fag_dataset
from util.argparser import get_args_parser, save_args
from util.mixup import mosaic
from datasets.transforms import resize


def mixup_resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return h, w

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return oh, ow

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    transform = T.Resize(size)
    rescaled_image = transform(image)

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size()[1:], image.size()[1:]))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"].to('cuda')
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height]).to('cuda')
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    return rescaled_image, target


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

    # print(dataset_train.annotations[0])
    # hoi_list = json.load(open(os.path.join("./data/annotations/", 'hoi_list_new.json'), 'r'))
    # with open("./data/annotations/hoi_id_to_num.json", "r") as file:
    #     hoi_rare_mapping = json.load(file)
    # count_dict = defaultdict()
    # rare_img_list = []  # boolean list
    # for i, anno in tqdm(enumerate(dataset_train.annotations)):
    #     is_rare = False
    #     hoi_label = []
    #     # print(anno['hoi_annotation'])
    #     for hoi in anno['hoi_annotation']:
    #         hoi_id = hoi['hoi_category_id'] - 1
    #         str_id = hoi_list[hoi_id]['id']
    #         hoi_dict = hoi_rare_mapping[str_id]
    #         hoi_label.append({
    #             "id": str_id,
    #             **hoi_dict,
    #         })
    #
    #         tmp = count_dict.get(hoi_id, 0)
    #         count_dict[hoi_id] = tmp + 1
    #
    #         if hoi_dict['rare'] == True:
    #             is_rare = True
    #
    #     rare_img_list.append(is_rare)

    # for i, (img, anno) in enumerate(dataset_train):
    #     print(img.size(), anno['orig_size'], anno['verb_labels'].size(),
    #           anno['hoi_labels'].size(), torch.sum(anno['hoi_labels'], -1))
    #     if i >= 20:
    #         break
    img, anno = dataset_train[0]
    # print(img.size())
    # batch = next(iter(data_loader_train))
    # samples, targets = batch
    # targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
    # # print(sum(len(t['hoi_labels']) for t in targets))
    # for t1, t2 in zip(targets[:8], targets[8:]):
    #     # print(t2['boxes'])
    #     image, t2 = mixup_resize(torch.rand(3, 544, 839), t2, (640, 480))
    #     # print(t2['boxes'])
    #     # for k, v in t1.items():
    #     #     if torch.is_tensor(v):
    #     #         print(k, v.size())
    #     # lamb = np.random.beta(2.0, 2.0)
    #     # new_obj_labels = torch.cat([t1['verb_labels']*lamb, t2['verb_labels']*(1-lamb)])
    #     # new_hoi_labels = torch.cat([t1['hoi_labels']*lamb, t2['hoi_labels']*(1-lamb)])
    #     break

    # def box_xyxy_to_cxcywh(x):
    #     x0, y0, x1, y1 = x
    #     b = [(x0 + x1) / 2, (y0 + y1) / 2,
    #          (x1 - x0), (y1 - y0)]
    #     return b
    # src_img = []
    # src_anno = []
    # src_size = []
    # for i, (img, anno) in enumerate(dataset_train):
    #     img_anno = dataset_train.annotations[i]
    #     img = Image.open(dataset_train.img_folder / img_anno['file_name']).convert('RGB')
    #     w, h = img.size
    #     transform = transforms.Compose([transforms.ToTensor()])
    #     if i >= 4:
    #         break
    #     # src_img.append(transform(img))
    #     src_img.append(np.array(img))
    #     src_size.append((h, w))
    #     for annos in img_anno['annotations']:
    #         src_anno.append({
    #             "bbox": box_xyxy_to_cxcywh(annos['bbox']),
    #             "cls": dataset_train._valid_obj_ids[annos['category_id']]
    #         })
    # x = torch.randn(2, 3)
    # tgt_img, tgt_anno = mosaic(src_img, src_anno, src_size)
    # tgt_img, tgt_anno = mosaic(src_img, src_anno, [480, 640])
    # print(src_anno)

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
