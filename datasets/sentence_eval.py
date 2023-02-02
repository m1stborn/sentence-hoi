# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import cv2
import json
import os
from collections import defaultdict

import numpy as np
import torch

from util.topk import top_k
from .hico_text_label import hico_text_label


def voc_ap(rec, prec):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.
    return ap


def cal_prec(rec, prec, t=0.8):
    if np.sum(rec >= t) == 0:
        p = 0
    else:
        p = np.max(prec[rec >= t])
    return p


class SentenceEvaluator:
    def __init__(self, sen_pred, gts, rare_triplets, non_rare_triplets, args):
        """
        :param sen_pred: List[dict] -> dict {
            "pred_hoi_label": 245, "pred_sentence_score": 0.9964583, "pred_sentence": "a photo of a person sitting on a bench"
        }
        """
        self.overlap_iou = 0.5
        self.max_hois = 100

        self.zero_shot_type = args.zero_shot_type

        self.use_nms_filter = args.use_nms_filter
        self.thres_nms = args.thres_nms
        self.nms_alpha = args.nms_alpha
        self.nms_beta = args.nms_beta

        self.use_score_thres = False
        self.thres_score = 1e-5

        self.use_soft_nms = False
        self.soft_nms_sigma = 0.5
        self.soft_nms_thres_score = 1e-11

        self.rare_triplets = rare_triplets
        self.non_rare_triplets = non_rare_triplets

        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)
        # self.gt_triplets = []
        self.gt_labels = []

        # self.preds = []
        self.sen_pred = sen_pred
        self.hico_triplet_labels = list(hico_text_label.keys())
        self.hoi_obj_list = []

        self.dev = args.dev

        # ----------------------------------
        self.text2triplet = {v: k for k, v in hico_text_label.items()}
        self.pair2text = hico_text_label
        self.text_list = list(self.pair2text.values())
        self.text2idx = {text:  self.text_list.index(text) for text in self.pair2text.values()}

        self.gts = []

        for i, img_gts in enumerate(gts):
            filename = img_gts['filename']

            # img_gts = {k: v.to('cpu').numpy() for k, v in img_gts.items() if k != 'id' and k != 'filename'}
            # img_gts = {k: v.to('cpu').numpy() for k, v in img_gts.items() if torch.is_tensor(v)}

            # bbox_anns = [{'bbox': list(bbox), 'category_id': label} for bbox, label in
            #              zip(img_gts['boxes'], img_gts['labels'])]

            # hoi_anns = [{'subject_id': hoi[0],
            #              'object_id': hoi[1]}
            #             for hoi in img_gts['hois']]

            self.gts.append({
                'filename': filename,
                # 'annotations': bbox_anns,
                # 'hoi_annotation': hoi_anns,
                'hoi_sentence': img_gts['hoi_sentence']
            })
            for sentence in self.gts[-1]['hoi_sentence']:
                # triplet = sentence['category_id']
                # hoi_lagel = self.text2triplet.get(sentence, (0, 0))

                # TODO: unseen = -1
                hoi_lagel = self.text2idx.get(sentence, -1)
                if hoi_lagel not in self.gt_labels:
                    self.gt_labels.append(hoi_lagel)

                self.sum_gts[hoi_lagel] += 1

            # if self.dev:
            #     break

        # with open(f"{args.output_dir}/{args.json_file}", 'w') as f:
        #     result = json.loads(str({'preds': self.preds, 'gts': self.gts}).replace("\'", "\""))
        #     json.dump(result, f)

        # with open(f"{args.output_dir}/{args.json_file}", 'w') as f:
        #     f.write(json.dumps(str({'preds': self.preds, 'gts': self.gts})))

        # with open(f"{args.output_dir}/{args.json_file}", "w") as file:
        #     json.dump({'preds': self.preds, 'gts': self.gts}, file)

        print(len(self.sen_pred))
        print(len(self.gts))
        # print(f"sentence_eval: {self.gts}")

    def evaluate(self):
        for img_pred, img_gts in zip(self.sen_pred, self.gts):
            if len(img_pred) == 0:
                continue

            if len(img_gts['hoi_sentence']) != 0:
                self.compute_fp_tp(img_pred, img_gts)
            else:
                for pred_objs in img_pred['hoi_prediction']:
                    hoi_label = pred_objs['pred_hoi_label']
                    if hoi_label not in self.gt_labels:
                        continue
                    self.tp[hoi_label].append(0)
                    self.fp[hoi_label].append(1)
                    # self.score[hoi_label].append(pred_objs['score'])

        return self.compute_map()

    def compute_fp_tp(self, img_pred, img_gts):
        unique_pred_sentence = defaultdict(dict)  # 'hoi_label' : pred_obj
        # TODO: check needed?
        hoi_predictions = img_pred['hoi_prediction']
        hoi_predictions.sort(key=lambda k: (k.get('score', 0)), reverse=True)  # descending

        for pred_objs in hoi_predictions:
            pred_sentence = pred_objs['pred_sentence']
            if pred_sentence not in unique_pred_sentence:
                unique_pred_sentence[pred_sentence] = pred_objs

            # if hoi_label not in self.gt_triplets:
            #     continue
            # self.score[hoi_label].append(pred_objs['score'])

        for gt_sentence in img_gts['hoi_sentence']:

            hoi_label = self.text2idx.get(gt_sentence, -1)
            if gt_sentence in unique_pred_sentence.keys():
                self.tp[hoi_label].append(1)
                self.fp[hoi_label].append(0)
            else:
                self.fp[hoi_label].append(1)
                self.tp[hoi_label].append(0)

        return

    def compute_map(self):
        # print("compute_map", self.tp, self.fp)
        ap = defaultdict(lambda: 0)
        # rare_ap = defaultdict(lambda: 0)
        # non_rare_ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        for hoi_label in self.gt_labels:
            sum_gts = self.sum_gts[hoi_label]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[hoi_label]))
            fp = np.array((self.fp[hoi_label]))
            if len(tp) == 0:
                ap[hoi_label] = 0
                max_recall[hoi_label] = 0
                # if triplet in self.rare_triplets:
                #     rare_ap[triplet] = 0
                # elif triplet in self.non_rare_triplets:
                #     non_rare_ap[triplet] = 0
                # else:
                #     print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
                continue

            # score = np.array(self.score[triplet])
            # sort_inds = np.argsort(-score)
            # fp = fp[sort_inds]
            # tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gts
            prec = tp / (fp + tp)
            # ap[triplet] = self.cal_prec(rec, prec)
            ap[hoi_label] = voc_ap(rec, prec)
            max_recall[hoi_label] = np.amax(rec)
            # orignal_triplet = self.hico_triplet_labels[hoi_label]
            # orignal_triplet = (0, orignal_triplet[1], orignal_triplet[0])
            # if orignal_triplet in self.rare_triplets:
            #     rare_ap[hoi_label] = ap[hoi_label]
            # elif orignal_triplet in self.non_rare_triplets:
            #     non_rare_ap[hoi_label] = ap[hoi_label]
            # else:
            #     print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
        m_ap = np.mean(list(ap.values()))
        # m_ap_rare = np.mean(list(rare_ap.values()))
        # m_ap_non_rare = np.mean(list(non_rare_ap.values()))
        m_max_recall = np.mean(list(max_recall.values()))

        print('--------------------')
        print(f'mAP full: {m_ap}  mean max recall: {m_max_recall}')
        return_dict = {'mAP': m_ap,
                       'mean max recall': m_max_recall}

        # if self.zero_shot_type == "default":
        #     print('mAP full: {} mAP rare: {}  mAP non-rare: {}  mean max recall: {}'.format(m_ap, m_ap_rare,
        #                                                                                     m_ap_non_rare,
        #                                                                                     m_max_recall))
        #     return_dict = {'mAP': m_ap, 'mAP rare': m_ap_rare, 'mAP non-rare': m_ap_non_rare,
        #                    'mean max recall': m_max_recall}
        #
        # elif self.zero_shot_type == "unseen_object":
        #     print('mAP full: {} mAP unseen-obj: {}  mAP seen-obj: {}  mean max recall: {}'.format(m_ap, m_ap_rare,
        #                                                                                           m_ap_non_rare,
        #                                                                                           m_max_recall))
        #     return_dict = {'mAP': m_ap, 'mAP unseen-obj': m_ap_rare, 'mAP seen-obj': m_ap_non_rare,
        #                    'mean max recall': m_max_recall}
        #
        # else:
        #     print('mAP full: {} mAP unseen: {}  mAP seen: {}  mean max recall: {}'.format(m_ap, m_ap_rare,
        #                                                                                   m_ap_non_rare,
        #                                                                                   m_max_recall))
        #     return_dict = {'mAP': m_ap, 'mAP unseen': m_ap_rare, 'mAP seen': m_ap_non_rare,
        #                    'mean max recall': m_max_recall}

        print('--------------------')

        return return_dict
