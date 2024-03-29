"""
HICO detection dataset.
"""
import os
import random
from pathlib import Path

import torchvision.transforms as t
from PIL import Image
import json
from collections import defaultdict
import numpy as np

import torch
import torch.utils.data
# import clip

import datasets.transforms as T
from .hico_text_label import hico_text_label, hico_unseen_index


class HICODetection(torch.utils.data.Dataset):
    def __init__(self, img_set, img_folder, anno_file, clip_feats_folder, transforms, num_queries, args):
        self.rare_triplets = None
        self.non_rare_triplets = None
        self.correct_mat = None
        self.img_set = img_set
        self.img_folder = img_folder
        # self.clip_feates_folder = clip_feats_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

        self.num_queries = num_queries

        self.unseen_index = hico_unseen_index.get(args.zero_shot_type, [])
        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        self._valid_verb_ids = list(range(1, 118))

        self.pair2text = hico_text_label
        self.text_label_ids = list(self.pair2text.keys())

        self.num_object = len(self._valid_obj_ids)
        self.num_action = len(self._valid_verb_ids)
        self.num_hoi = len(hico_text_label)

        if img_set == 'train' and len(self.unseen_index) != 0 and args.del_unseen:
            tmp = []
            for idx, k in enumerate(self.text_label_ids):
                if idx in self.unseen_index:
                    continue
                else:
                    tmp.append(k)
            self.text_label_ids = tmp

        if img_set == 'train':
            self.ids = []
            for idx, img_anno in enumerate(self.annotations):
                new_img_anno = []
                skip_pair = []
                for hoi in img_anno['hoi_annotation']:
                    if hoi['hoi_category_id'] - 1 in self.unseen_index:
                        skip_pair.append((hoi['subject_id'], hoi['object_id']))
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(
                            img_anno['annotations']):
                        new_img_anno = []
                        break
                    if (hoi['subject_id'], hoi['object_id']) not in skip_pair:
                        new_img_anno.append(hoi)
                if len(new_img_anno) > 0:
                    self.ids.append(idx)
                    img_anno['hoi_annotation'] = new_img_anno
        else:
            self.ids = list(range(len(self.annotations)))
        print("{} contains {} images".format(img_set, len(self.ids)))

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # _, self.clip_preprocess = clip.load(args.clip_model, device)
        self.mixup = False
        if args.mixup and img_set == 'train':
            self.mixup = args.mixup
            self.mixup_prob = 0.9
            self.mixup_alpha = 2
            with open(f"{args.hoi_path}/annotations/bg_image_idx.json", 'r') as f:
                bg_data = json.load(f)
            self.bg_image_filename = bg_data['bg_image_filename']
            self.mixup_keywords = ['boxes', 'labels', 'hoi_sentence', 'valid_pairs',
                                   'obj_labels', 'verb_labels', 'hoi_labels', 'sub_boxes', 'obj_boxes']
            with open(f"./data/annotations/train_img_contains_rare_hoi.json", 'r') as f:
                self.img_rare_list = json.load(f)
                self.rare_idx_list = [i for i, x in enumerate(self.img_rare_list) if x]
                # print(self.rare_idx_list)

    def __len__(self):
        return len(self.ids)

    def get_mixup(self, idx, target_size=None):
        img_anno = self.annotations[self.ids[idx]]

        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        w, h = img.size

        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        if self.img_set == 'train':
            # Add index for confirming which boxes are kept after image transformation
            classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in
                       enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {'orig_size': torch.as_tensor([int(h), int(w)]), 'size': torch.as_tensor([int(h), int(w)])}

        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        # Box relative annotation
        target['boxes'] = boxes
        target['labels'] = classes
        target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
        target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        mixup_transform = T.Compose([
            T.TargetResize(target_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img, target = mixup_transform(img, target)

        kept_box_indices = [label[0] for label in target['labels']]

        target['labels'] = target['labels'][:, 1]

        obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
        sub_obj_pairs = []
        hoi_labels = []
        hoi_sentences = []
        valid_pair = []

        for hoi in img_anno['hoi_annotation']:
            if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                continue
            verb_obj_pair = (self._valid_verb_ids.index(hoi['category_id']),
                             target['labels'][kept_box_indices.index(hoi['object_id'])])
            if verb_obj_pair not in self.text_label_ids:
                continue

            pair = (self._valid_verb_ids.index(hoi['category_id']),
                    target['labels'][kept_box_indices.index(hoi['object_id'])].item())
            sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
            if sub_obj_pair in sub_obj_pairs:
                verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                hoi_labels[sub_obj_pairs.index(sub_obj_pair)][self.text_label_ids.index(verb_obj_pair)] = 1
            else:
                sub_obj_pairs.append(sub_obj_pair)
                obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                hoi_label = [0] * len(self.text_label_ids)
                hoi_label[self.text_label_ids.index(verb_obj_pair)] = 1
                verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                verb_labels.append(verb_label)
                hoi_labels.append(hoi_label)
                sub_boxes.append(sub_box)
                obj_boxes.append(obj_box)

                hoi_sentences.append(self.pair2text[pair])
                valid_pair.append(pair)
        target['filename'] = img_anno['file_name']

        target['hoi_sentence'] = hoi_sentences if len(hoi_sentences) != 0 else ['A photo of a person.']
        target['valid_pairs'] = valid_pair

        if len(sub_obj_pairs) == 0:
            target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
            target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
            target['hoi_labels'] = torch.zeros((0, len(self.text_label_ids)), dtype=torch.float32)
            target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
        else:
            target['obj_labels'] = torch.stack(obj_labels)
            target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
            target['hoi_labels'] = torch.as_tensor(hoi_labels, dtype=torch.float32)
            target['sub_boxes'] = torch.stack(sub_boxes)
            target['obj_boxes'] = torch.stack(obj_boxes)

        return img, target

    def __getitem__(self, idx):
        img_anno = self.annotations[self.ids[idx]]

        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        w, h = img.size

        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        if self.img_set == 'train':
            # Add index for confirming which boxes are kept after image transformation
            classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in
                       enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {'orig_size': torch.as_tensor([int(h), int(w)]), 'size': torch.as_tensor([int(h), int(w)])}
        if self.img_set == 'train':
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            # Box relative annotation
            target['boxes'] = boxes
            target['labels'] = classes
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            bg_size = None
            if self._transforms is not None:
                img_0, target_0 = self._transforms[0](img, target)
                img, target = self._transforms[1](img_0, target_0)
                bg_size = (img.size(2), img.size(1))  # reverse

            kept_box_indices = [label[0] for label in target['labels']]

            target['labels'] = target['labels'][:, 1]

            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            sub_obj_pairs = []
            hoi_labels = []
            hoi_sentences = []
            valid_pair = []
            # hoi_sentences_candidate = []

            # For hoi.csv generation
            hoi_pair = []

            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                    continue
                verb_obj_pair = (self._valid_verb_ids.index(hoi['category_id']),
                                 target['labels'][kept_box_indices.index(hoi['object_id'])])
                if verb_obj_pair not in self.text_label_ids:
                    continue

                pair = (self._valid_verb_ids.index(hoi['category_id']),
                        target['labels'][kept_box_indices.index(hoi['object_id'])].item())

                hoi_pair.append(pair)
                # hoi_sentences_candidate.append(self.pair2text[pair])

                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                # hoi_pair.append(sub_obj_pair)
                if sub_obj_pair in sub_obj_pairs:
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                    hoi_labels[sub_obj_pairs.index(sub_obj_pair)][self.text_label_ids.index(verb_obj_pair)] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    hoi_label = [0] * len(self.text_label_ids)
                    hoi_label[self.text_label_ids.index(verb_obj_pair)] = 1
                    verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                    verb_labels.append(verb_label)
                    hoi_labels.append(hoi_label)
                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)

                    hoi_sentences.append(self.pair2text[pair])
                    valid_pair.append(pair)
            target['filename'] = img_anno['file_name']

            target['hoi_sentence'] = hoi_sentences if len(hoi_sentences) != 0 else ['A photo of a person.']
            target['hoi_pairs'] = hoi_pair
            target['valid_pairs'] = valid_pair

            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['hoi_labels'] = torch.zeros((0, len(self.text_label_ids)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['hoi_labels'] = torch.as_tensor(hoi_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)

            # mixup
            # if self.mixup and not self.img_rare_list[idx]:
            if self.mixup and not self.img_rare_list[idx] and torch.rand(1) > self.mixup_prob:
                lamb = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                mixup_idx = random.choice(self.rare_idx_list)
                # print("mixup_idx", mixup_idx)
                mixup_img, mixup_anno = self.get_mixup(mixup_idx, bg_size)

                # mixup images
                img = img * lamb + (1 - lamb) * mixup_img
                # mixup annotations
                # num_extend_bboxes = len(mixup_anno['boxes'])
                target['boxes'] = torch.cat([target['boxes'], mixup_anno['boxes']])
                target['labels'] = torch.cat([target['labels'], mixup_anno['labels']])
                target['sub_boxes'] = torch.cat([target['sub_boxes'], mixup_anno['sub_boxes']])
                target['obj_boxes'] = torch.cat([target['obj_boxes'], mixup_anno['obj_boxes']])

                target['hoi_sentence'] += mixup_anno['hoi_sentence']
                target['valid_pairs'] += mixup_anno['valid_pairs']
                target['obj_labels'] = torch.cat([target['obj_labels'], mixup_anno['obj_labels']])

                target['verb_labels'] = torch.cat([target['verb_labels']*lamb, mixup_anno['verb_labels']*(1-lamb)])
                target['hoi_labels'] = torch.cat([target['hoi_labels']*lamb, mixup_anno['hoi_labels']*(1-lamb)])

        else:
            target['filename'] = img_anno['file_name']
            target['boxes'] = boxes
            target['labels'] = classes
            target['id'] = idx

            if self._transforms is not None:
                img, _ = self._transforms(img, None)

            hois = []
            hoi_sentences = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))

                pair = (self._valid_verb_ids.index(hoi['category_id']), hoi['object_id'])
                hoi_sentences.append(self.pair2text.get(pair, "A photo of a person."))

            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)
            target['hoi_sentence'] = hoi_sentences

        return img, target

    def set_rare_hois(self, anno_file):
        with open(anno_file, 'r') as f:
            annotations = json.load(f)

        if len(self.unseen_index) == 0:
            # no unseen categoruy, use rare to evaluate
            counts = defaultdict(lambda: 0)
            for img_anno in annotations:
                hois = img_anno['hoi_annotation']
                bboxes = img_anno['annotations']
                for hoi in hois:
                    triplet = (self._valid_obj_ids.index(bboxes[hoi['subject_id']]['category_id']),
                               self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id']),
                               self._valid_verb_ids.index(hoi['category_id']))
                    counts[triplet] += 1
            self.rare_triplets = []
            self.non_rare_triplets = []
            for triplet, count in counts.items():
                if count < 10:
                    self.rare_triplets.append(triplet)
                else:
                    self.non_rare_triplets.append(triplet)
            print("rare:{}, non-rare:{}".format(len(self.rare_triplets), len(self.non_rare_triplets)))
        else:
            self.rare_triplets = []
            self.non_rare_triplets = []
            for img_anno in annotations:
                hois = img_anno['hoi_annotation']
                bboxes = img_anno['annotations']
                for hoi in hois:
                    triplet = (self._valid_obj_ids.index(bboxes[hoi['subject_id']]['category_id']),
                               self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id']),
                               self._valid_verb_ids.index(hoi['category_id']))
                    if hoi['hoi_category_id'] - 1 in self.unseen_index:
                        self.rare_triplets.append(triplet)
                    else:
                        self.non_rare_triplets.append(triplet)
            print("unseen:{}, seen:{}".format(len(self.rare_triplets), len(self.non_rare_triplets)))

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)


# Add color jitter to coco transforms
def make_hico_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640]
    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return [T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=900),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=900),
                ]))]
        ),
            normalize
        ]

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    paths = {
        'train': (root / 'images' / 'train2015', root / 'annotations' / 'trainval_hico.json',
                  root / 'clip_feats_pool' / 'train2015'),
        'val': (
            root / 'images' / 'test2015', root / 'annotations' / 'test_hico.json',
            root / 'clip_feats_pool' / 'test2015')
    }
    correct_mat_path = root / 'annotations' / 'corre_hico.npy'

    img_folder, anno_file, clip_feats_folder = paths[image_set]
    dataset = HICODetection(image_set, img_folder, anno_file, clip_feats_folder,
                            transforms=make_hico_transforms(image_set),
                            num_queries=args.num_queries, args=args)
    if image_set == 'val':
        dataset.set_rare_hois(paths['train'][1])
        dataset.load_correct_mat(correct_mat_path)
    return dataset
