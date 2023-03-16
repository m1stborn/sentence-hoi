import json
import os

import numpy as np

from datasets.hico_fag_eval import (
    compute_ap,
    compute_pr,
    compute_center_distacne,
    compute_large_area,
    compute_iou,
    dump_json_object,
    match_hoi
)
from util.topk import top_k
from .hico_text_label import hico_text_label


def match_bbox(pred_bboxes, gt_bboxes):
    # is_match = False
    remaining_bboxes = [gt_bbox for gt_bbox in gt_bboxes]
    # assert remaining_bboxes == gt_bboxes
    # print("start", len(remaining_bboxes))
    for j, pred_bbox in enumerate(pred_bboxes):
        for i, gt_bbox in enumerate(remaining_bboxes):
            iou = compute_iou(pred_bbox, gt_bbox)
            if iou > 0.5:
                try:
                    remaining_bboxes.remove(gt_bbox)
                except ValueError:
                    print(remaining_bboxes, gt_bbox)
            if len(remaining_bboxes) == 0:
                return []

    return remaining_bboxes


def match_bbox_category(pred_bboxes, pred_class, gt_bboxes, gt_class):
    match_and_correct = []  # bool
    for j, pred_bbox in enumerate(pred_bboxes):
        correct = False
        any_match = False
        for i, gt_bbox in enumerate(gt_bboxes):
            iou = compute_iou(pred_bbox, gt_bbox)
            if iou > 0.5:
                any_match = True
                if pred_class[j] == gt_class[i]:
                    correct = True
                    match_and_correct.append(correct)
                    break
        if not correct and any_match:
            match_and_correct.append(False)

    return match_and_correct


def match_pair(bboxes, pred_pairs, gt_pairs):
    """
    exam each pair correctness
    """
    collate = {label: 0 for h_id, o_id, label in pred_pairs}

    match_and_correct = []  # pairwise match
    ap_dict = {}
    for i, (h_id, o_id, label) in enumerate(pred_pairs):
        any_match = False
        correct = False
        for j, gt_det in enumerate(gt_pairs):
            human_iou = compute_iou(bboxes[h_id]['bbox'], gt_det['human_box'])
            if human_iou > 0.5:
                object_iou = compute_iou(bboxes[o_id]['bbox'], gt_det['object_box'])
                if object_iou > 0.5:
                    any_match = True
                    if label == gt_det['category_id']:
                        correct = True
                        # match_and_correct.append(True)
                        if label in ap_dict:
                            ap_dict[label].append(True)
                        else:
                            ap_dict[label] = [True]
                        break  # continue search for match pair
        collate[label] += 1
        if any_match:
            match_and_correct.append(True)
        if not correct and any_match:
            # match_and_correct.append(False)
            if label in ap_dict:
                ap_dict[label].append(False)
            else:
                ap_dict[label] = [False]

    print('match_pair', collate)
    # print(ap_dict)

    # print(match_and_correct)
    # print(len(match_and_correct), len(pred_pairs))
    # total_sum_ap = {k: np.mean(v) if len(v) else 0.0 for k, v in ap_dict.items()}
    # print(total_sum_ap)
    return ap_dict


class HicoEvalBbox:
    def __init__(self, predictions, gts, dataset_path, out_dir, epoch, bins_num=10, use_nms=True, nms_thresh=0.5):
        self.out_dir = out_dir
        self.epoch = epoch
        self.bins_num = bins_num
        self.bins = np.linspace(0, 1.0, self.bins_num + 1)
        self.compute_extra = {'distance': compute_center_distacne, 'area': compute_large_area}
        self.extra_keys = list(self.compute_extra.keys())
        self.ap_compute_set = {k: v for k, v in
                               zip(self.extra_keys, [self._ap_compute_set() for i in range(len(self.extra_keys))])}
        self.img_size_info = {}
        self.img_folder = os.path.join(dataset_path, 'images/test2015')
        self.anno_path = os.path.join(dataset_path, "annotations")
        self.annotations = self.load_gt_dets()
        self.hoi_list = json.load(open(os.path.join(self.anno_path, 'hoi_list_new.json'), 'r'))
        self.file_name_to_obj_cat = json.load(open(os.path.join(self.anno_path, 'file_name_to_obj_cat.json'), "r"))
        self.nms_thresh = nms_thresh

        self.global_ids = self.annotations.keys()
        self.hoi_id_to_num = json.load(open(os.path.join(self.anno_path, 'hoi_id_to_num.json'), "r"))
        self.rare_id_json = [key for key, item in self.hoi_id_to_num.items() if item['rare']]

        self.correct_mat = np.load(os.path.join(self.anno_path, 'corre_hico.npy'))
        self.valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                              14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                              24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                              37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                              48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                              58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                              72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                              82, 84, 85, 86, 87, 88, 89, 90)  # Total 80
        self.valid_verb_ids = list(range(1, 118))

        self.pred_anno = {}
        self.preds_t = []
        self.thres_nms = 0.7
        self.use_nms = use_nms
        self.max_hois = 100

        self.hoi_obj_list = []
        self.hico_triplet_labels = list(hico_text_label.keys())
        for hoi_pair in self.hico_triplet_labels:
            self.hoi_obj_list.append(hoi_pair[1])

        # Check label
        self.mapping = {}
        for idx, (item, (v_id, o_id)) in enumerate(zip(self.hoi_list, self.hico_triplet_labels)):
            # print(f"obj {o_id}, verb {v_id}, {item['object_cat'], item['verb_id'] } ")
            assert item['object_index'] == o_id
            assert item['verb_id'] - 1 == v_id
            self.mapping[idx] = item["id"]

        print("Convert preds... HicoEvalBbox")
        count = 0
        for img_preds, img_gts in zip(predictions, gts):
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
            bboxes = [{'bbox': bbox, 'category_id': self.valid_obj_ids[label]} for bbox, label in
                      zip(img_preds['boxes'], img_preds['labels'])]
            # hoi_scores = img_preds['verb_scores']  # 100, 117
            obj_scores = img_preds['obj_scores'] * img_preds['obj_scores']
            hoi_scores = img_preds['hoi_scores'] + obj_scores[:, self.hoi_obj_list]

            verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))  # [0 1 ... 117 1 2 ...]
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

            hoi_scores = hoi_scores.ravel()  # 117*100
            verb_labels = verb_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()

            topk_hoi_scores = top_k(list(hoi_scores), self.max_hois)
            topk_indexes = np.array([np.where(hoi_scores == score)[0][0] for score in topk_hoi_scores])

            if len(subject_ids) > 0:
                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score}
                        for subject_id, object_id, category_id, score in
                        zip(subject_ids[topk_indexes],
                            object_ids[topk_indexes],
                            verb_labels[topk_indexes],
                            topk_hoi_scores)]
                hois = hois[:self.max_hois]
            else:
                hois = []

            count += 1

            # filename = img_gts["file_name"].split('.')[0]
            filename = img_gts['filename'].split('.')[0]
            self.preds_t.append({
                'filename': filename,
                'predictions': bboxes,
                'hoi_prediction': hois
            })
        if self.use_nms:
            self.preds_t = self.triplet_nms_filter(self.preds_t)

        anno_list = json.load(open(os.path.join(self.anno_path, 'test_anno_list.json'), "r"))
        remain_bbox_json = {}
        total_human_bbox = 0
        total_object_bbox = 0
        total_remain_human_bbox = 0
        total_remain_object_bbox = 0
        match_and_corrects = []
        ap_dict = {e['id']: [] for e in self.hoi_list}
        hoi2obj = {e['id']: e['object_cat'] for e in self.hoi_list}
        # hoi2 = {e['id']: e['object_cat'] for e in self.hoi_list}

        predictions_dict = {pred['filename']: pred for pred in self.preds_t}
        anno_dict = {gt['global_id']: gt for gt in anno_list}
        gt_dets = self.load_gt_dets_for_matching()
        for filename, pred in predictions_dict.items():
            # print(pred['hoi_prediction'][:3])
            # break
            gt = anno_dict[filename]

            collate_gt_obj_pair = [(b, pair['id']) for pair in gt['hois'] for b in pair['object_bboxes']]  # list of bbox
            collate_gt_human = [b for pair in gt['hois'] for b in pair['human_bboxes']]  # list of bbox
            collate_gt_object = [b for b, _ in collate_gt_obj_pair]
            collate_gt_obj_class = [hoi2obj[idx] for _, idx in collate_gt_obj_pair]
            assert len(collate_gt_object) == len(collate_gt_obj_class)

            collate_pred_human = [p['bbox'] for p in pred['predictions'] if p['category_id'] == 1]  # list of bbox
            collate_pred_object = [p['bbox'] for p in pred['predictions'] if p['category_id'] != 1]  # list of bbox
            collate_pred_obj_class = [p['category_id'] for p in pred['predictions'] if p['category_id'] != 1]
            assert len(collate_pred_obj_class) == len(collate_pred_object)

            # Accuracy of detection object
            match_and_correct = match_bbox_category(collate_pred_object, collate_pred_obj_class,
                                                    collate_gt_object, collate_gt_obj_class)
            match_and_corrects.extend(match_and_correct)

            # Recall of bbox
            total_object_bbox += len(collate_gt_object)
            total_human_bbox += len(collate_gt_human)
            remain_human_bbox = match_bbox(collate_pred_human, collate_gt_human)
            remain_object_bbox = match_bbox(collate_pred_object, collate_gt_object)
            total_remain_object_bbox += len(remain_object_bbox)
            total_remain_human_bbox += len(remain_human_bbox)

            remain_bbox_json[pred['filename']] = {
                "human_bboxes": remain_human_bbox,
                "object_bboxes": remain_object_bbox
            }

            # pair wise bbox
            collate_pred_pair = [(p['subject_id'], p['object_id'], self.hoi_list[p['category_id']]['id'])
                                 for p in pred['hoi_prediction']]
            ap = match_pair(pred['predictions'], collate_pred_pair, gt_dets[filename])
            for k, v in ap.items():
                ap_dict[k].extend(v)
            break
        total_sum_ap_dict = {k: f"{np.sum(v)}/{len(v)}" for k, v in ap_dict.items() if len(v) > 0}
        # print(total_sum_ap_dict)
        # total_sum_ap = [np.mean(v) if len(v) else 0.0 for k, v in ap_dict.items()]
        # print(f"mean ap: {np.mean(total_sum_ap)}")
        # print(f"detect obj accuracy: {np.mean(match_and_corrects)*100:.2f} %")
        # print(f"detect human box {100*total_remain_human_bbox/total_human_bbox:.2f} % "
        #       f"({total_remain_human_bbox}/{total_human_bbox})")
        # print(f"detect object box: {100*total_remain_object_bbox/total_object_bbox:.2f} % "
        #       f"({total_remain_object_bbox}/{total_object_bbox})")

        # with open(f"{self.out_dir}/remain_bbox.json", "w", encoding="utf-8") as file:
        #     json.dump(remain_bbox_json, file, ensure_ascii=False, indent=4)

        for preds_i in self.preds_t:

            # convert
            global_id = preds_i["filename"]
            self.pred_anno[global_id] = {}
            hois = preds_i["hoi_prediction"]
            bboxes = preds_i["predictions"]
            for hoi in hois:
                obj_id = bboxes[hoi['object_id']]['category_id']
                obj_bbox = bboxes[hoi['object_id']]['bbox']
                sub_bbox = bboxes[hoi['subject_id']]['bbox']
                score = hoi['score']
                verb_id = hoi['category_id']  # 0-599
                hoi_id = self.mapping[verb_id]
                assert int(hoi_id) > 0

                data = np.array([sub_bbox[0], sub_bbox[1], sub_bbox[2], sub_bbox[3],
                                 obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3],
                                 score]).reshape(1, 9)
                if hoi_id not in self.pred_anno[global_id]:
                    self.pred_anno[global_id][hoi_id] = np.empty([0, 9])

                self.pred_anno[global_id][hoi_id] = np.concatenate((self.pred_anno[global_id][hoi_id], data), axis=0)

        for k in list(self.pred_anno.keys()):
            if k != filename:
                self.pred_anno.pop(k)

        # print(self.pred_anno['HICO_test2015_00000001'].keys())
        num1 = {k: f"{len(v)}" for k, v in self.pred_anno['HICO_test2015_00000001'].items()}
        print(num1)
        o = self.eval_hoi('246', self.global_ids, self.annotations, self.pred_anno, self.out_dir)
        print(o)

        # for hoi_id in ap_dict.keys():
        #     o = self.eval_hoi(hoi_id, self.global_ids, self.annotations, self.pred_anno, self.out_dir)
        #     print(hoi_id, o)
        #     break

    def load_gt_dets_for_matching(self):
        # Load anno_list
        anno_list = json.load(open(os.path.join(self.anno_path, 'test_anno_list.json'), "r"))

        gt_dets = {}
        for anno in anno_list:
            global_id = anno['global_id']
            gt_dets[global_id] = []
            for hoi in anno['hois']:
                hoi_id = hoi['id']
                gt_dets[global_id] = []
                for human_box_num, object_box_num in hoi['connections']:
                    human_box = hoi['human_bboxes'][human_box_num]
                    object_box = hoi['object_bboxes'][object_box_num]
                    det = {
                        'human_box': human_box,
                        'object_box': object_box,
                        'category_id': hoi_id,
                    }
                    gt_dets[global_id].append(det)

        return gt_dets

    def load_gt_dets(self):
        # Load anno_list
        print('Loading anno_list.json ...')
        anno_list = json.load(open(os.path.join(self.anno_path, 'anno_list.json'), "r"))

        gt_dets = {}
        for anno in anno_list:
            if "test" not in anno['global_id']:
                continue

            global_id = anno['global_id']
            gt_dets[global_id] = {}
            img_h, img_w, _ = anno['image_size']
            self.img_size_info[global_id] = [img_h, img_w]
            for hoi in anno['hois']:
                hoi_id = hoi['id']
                gt_dets[global_id][hoi_id] = []
                for human_box_num, object_box_num in hoi['connections']:
                    human_box = hoi['human_bboxes'][human_box_num]
                    object_box = hoi['object_bboxes'][object_box_num]
                    det = {
                        'human_box': human_box,
                        'object_box': object_box,
                    }
                    gt_dets[global_id][hoi_id].append(det)

        return gt_dets

    def _ap_compute_set(self):
        out = {
            'y_true': [[] for _ in range(self.bins_num)],
            'y_score': [[] for _ in range(self.bins_num)],
            'npos': [0 for _ in range(self.bins_num)]
        }
        return out

    def evaluation_default(self):
        outputs = []
        for hoi in self.hoi_list:
            o = self.eval_hoi(hoi['id'], self.global_ids, self.annotations, self.pred_anno, self.out_dir)
            outputs.append(o)
            # break

        m_ap = {
            'AP': {},
            'mAP': 0,
            'invalid': 0,
            'mAP_rare': 0,
            'mAP_non_rare': 0,
        }
        map_ = 0
        map_rare = 0
        map_non_rare = 0
        count = 0
        count_rare = 0
        count_non_rare = 0
        for ap, hoi_id in outputs:
            m_ap['AP'][hoi_id] = ap
            if not np.isnan(ap):
                count += 1
                map_ += ap
                if hoi_id in self.rare_id_json:
                    count_rare += 1
                    map_rare += ap
                else:
                    count_non_rare += 1
                    map_non_rare += ap

        m_ap['mAP'] = map_ / count
        m_ap['invalid'] = len(outputs) - count
        m_ap['mAP_rare'] = map_rare / count_rare
        m_ap['mAP_non_rare'] = map_non_rare / count_non_rare

        # TODO: move outside, now save every epoch not "best" epoch/
        m_ap_json = os.path.join(
            self.out_dir,
            f'mAP_default.json')
        dump_json_object(m_ap, m_ap_json)

        print(f'APs have been saved to {self.out_dir}')
        return {"mAP_def": m_ap['mAP'], "mAP_def_rare": m_ap['mAP_rare'], "mAP_def_non_rare": m_ap['mAP_non_rare']}

    def eval_hoi(self, hoi_id, global_ids, gt_dets, pred_anno,
                 mode='default', obj_cate=None):
        y_true = []
        y_score = []
        det_id = []
        npos = 0
        # remaining_det = {}
        for global_id in global_ids:
            if mode == 'ko':
                if global_id + ".jpg" not in self.file_name_to_obj_cat:
                    continue
                obj_cats = self.file_name_to_obj_cat[global_id + ".jpg"]
                if int(obj_cate) not in obj_cats:
                    continue

            if hoi_id in gt_dets[global_id]:
                candidate_gt_dets = gt_dets[global_id][hoi_id]
            else:
                candidate_gt_dets = []

            npos += len(candidate_gt_dets)

            if global_id not in pred_anno or hoi_id not in pred_anno[global_id]:
                hoi_dets = np.empty([0, 9])
            else:
                hoi_dets = pred_anno[global_id][hoi_id]

            num_dets = hoi_dets.shape[0]
            # sort by score
            sorted_idx = [idx for idx, _ in sorted(
                zip(range(num_dets), hoi_dets[:, 8].tolist()),
                key=lambda x: x[1],
                reverse=True)]
            for i in sorted_idx:
                pred_det = {
                    'human_box': hoi_dets[i, :4],
                    'object_box': hoi_dets[i, 4:8],
                    'score': hoi_dets[i, 8]
                }
                # print(hoi_dets[i, 8])
                is_match, candidate_gt_dets = match_hoi(pred_det, candidate_gt_dets)
                y_true.append(is_match)
                y_score.append(pred_det['score'])
                det_id.append((global_id, i))
            if global_id == "HICO_test2015_00000001":
                print('eval_hoi', hoi_id, num_dets, y_true, y_score)

        # Compute PR
        precision, recall, mark = compute_pr(y_true, y_score, npos)
        if not mark:
            ap = 0
        else:
            ap = compute_ap(precision, recall)

        return ap, hoi_id

    def triplet_nms_filter(self, preds):
        preds_filtered = []
        for img_preds in preds:
            pred_bboxes = img_preds['predictions']
            pred_hois = img_preds['hoi_prediction']
            all_triplets = {}
            for index, pred_hoi in enumerate(pred_hois):
                triplet = str(pred_bboxes[pred_hoi['subject_id']]['category_id']) + '_' + \
                          str(pred_bboxes[pred_hoi['object_id']]['category_id']) + '_' + str(pred_hoi['category_id'])

                if triplet not in all_triplets:
                    all_triplets[triplet] = {'subs': [], 'objs': [], 'scores': [], 'indexes': []}
                all_triplets[triplet]['subs'].append(pred_bboxes[pred_hoi['subject_id']]['bbox'])
                all_triplets[triplet]['objs'].append(pred_bboxes[pred_hoi['object_id']]['bbox'])
                all_triplets[triplet]['scores'].append(pred_hoi['score'])
                all_triplets[triplet]['indexes'].append(index)

            all_keep_inds = []
            for triplet, values in all_triplets.items():
                subs, objs, scores = values['subs'], values['objs'], values['scores']
                keep_inds = self.pairwise_nms(np.array(subs), np.array(objs), np.array(scores))

                keep_inds = list(np.array(values['indexes'])[keep_inds])
                all_keep_inds.extend(keep_inds)

            preds_filtered.append({
                'filename': img_preds['filename'],
                'predictions': pred_bboxes,
                'hoi_prediction': list(np.array(img_preds['hoi_prediction'])[all_keep_inds])
            })

        return preds_filtered

    def pairwise_nms(self, subs, objs, scores):
        sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
        ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

        sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
        obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

        order = scores.argsort()[::-1]

        keep_inds = []
        while order.size > 0:
            i = order[0]
            keep_inds.append(i)

            sxx1 = np.maximum(sx1[i], sx1[order[1:]])
            syy1 = np.maximum(sy1[i], sy1[order[1:]])
            sxx2 = np.minimum(sx2[i], sx2[order[1:]])
            syy2 = np.minimum(sy2[i], sy2[order[1:]])

            sw = np.maximum(0.0, sxx2 - sxx1 + 1)
            sh = np.maximum(0.0, syy2 - syy1 + 1)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

            oxx1 = np.maximum(ox1[i], ox1[order[1:]])
            oyy1 = np.maximum(oy1[i], oy1[order[1:]])
            oxx2 = np.minimum(ox2[i], ox2[order[1:]])
            oyy2 = np.minimum(oy2[i], oy2[order[1:]])

            ow = np.maximum(0.0, oxx2 - oxx1 + 1)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

            ovr = sub_inter / sub_union * obj_inter / obj_union
            inds = np.where(ovr <= self.nms_thresh)[0]

            order = order[inds + 1]
        return keep_inds
