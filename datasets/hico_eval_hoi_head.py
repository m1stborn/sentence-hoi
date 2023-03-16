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


class HICOHoiHeadEvaluator:
    def __init__(self, preds, gts, dataset_path, out_dir, epoch, bins_num=10, use_nms=True, nms_thresh=0.5):
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
            self.mapping[idx] = item["id"]  # str id

        print("Convert preds... HICOHoiHeadEvaluator")
        count = 0
        for img_preds, img_gts in zip(preds, gts):
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

            # TODO: filter invalid object, verb pair
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
        # print(self.preds_t[0]['predictions'])
        if self.use_nms:
            self.preds_t = self.triplet_nms_filter(self.preds_t)

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
                # print(verb_id, hoi_id)
                assert int(hoi_id) > 0

                data = np.array([sub_bbox[0], sub_bbox[1], sub_bbox[2], sub_bbox[3],
                                 obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3],
                                 score]).reshape(1, 9)
                if hoi_id not in self.pred_anno[global_id]:
                    self.pred_anno[global_id][hoi_id] = np.empty([0, 9])

                self.pred_anno[global_id][hoi_id] = np.concatenate((self.pred_anno[global_id][hoi_id], data), axis=0)

        # with open(f"{self.out_dir}/result.json", 'w', encoding="utf-8") as f:
        #     print(self.pred_anno)
        #     result = json.loads(str({"preds": self.preds_t}).replace("\'", "\""))
        #     json.dump(result, f)

        # cur_key = list(self.pred_anno.keys())
        # for k in cur_key:
        #     if k != 'HICO_test2015_00000001':
        #         self.pred_anno.pop(k)
        # # print(self.pred_anno.keys())
        # # print(self.pred_anno['HICO_test2015_00000001'].keys())
        # self.verbose_list = list(self.pred_anno['HICO_test2015_00000001'].keys())
        # print(sorted(self.verbose_list))
        # # print(self.annotations['HICO_test2015_00000001'].keys())
        # total = 0
        # for k, v in self.pred_anno['HICO_test2015_00000001'].items():
        #     print(k, v.shape)
        #     total += v.shape[0]
        # print(total)

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

    def match_hoi_extra(self, pred_det, gt_dets, img_h, img_w):
        is_match = False
        remaining_gt_dets = [gt_det for gt_det in gt_dets]
        extra_info = {}
        for extra_i in self.extra_keys:
            extra_info[extra_i + '_pred'] = self.compute_extra[extra_i](pred_det['human_box'], pred_det['object_box'],
                                                                        img_h, img_w)
        for i, gt_det in enumerate(gt_dets):
            human_iou = compute_iou(pred_det['human_box'], gt_det['human_box'])
            if human_iou > 0.5:
                object_iou = compute_iou(pred_det['object_box'], gt_det['object_box'])
                if object_iou > 0.5:
                    is_match = True
                    del remaining_gt_dets[i]
                    for extra_i in self.extra_keys:
                        extra_info[extra_i + '_gt'] = self.compute_extra[extra_i](gt_det['human_box'],
                                                                                  gt_det['object_box'], img_h, img_w)
                    break

        return is_match, remaining_gt_dets, extra_info

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

    def evaluation_ko(self):
        outputs = []
        for hoi in self.hoi_list:
            o = self.eval_hoi(hoi['id'], self.global_ids, self.annotations,
                              self.pred_anno, mode="ko",
                              obj_cate=hoi['object_cat'])
            outputs.append(o)

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
        # print(count_rare, count_non_rare)
        m_ap['mAP_rare'] = map_rare / count_rare
        m_ap['mAP_non_rare'] = map_non_rare / count_non_rare

        m_ap_json = os.path.join(
            self.out_dir,
            f'epo_{self.epoch}_mAP_ko.json')
        dump_json_object(m_ap, m_ap_json)

        print(f'APs have been saved to {self.out_dir}')
        return {"mAP_ko": m_ap['mAP'], "mAP_ko_rare": m_ap['mAP_rare'], "mAP_ko_non_rare": m_ap['mAP_non_rare']}

    def evaluation_extra(self):
        for hoi in self.hoi_list:
            self.eval_extra(hoi['id'], self.global_ids, self.annotations, self.pred_anno)

        extra_ap = {k: v for k, v in
                    zip(self.extra_keys, [[{} for j in range(self.bins_num)] for i in range(len(self.extra_keys))])}
        for bins_i in range(self.bins_num):
            for extra_i in self.extra_keys:
                y_true = self.ap_compute_set[extra_i]['y_true'][bins_i]
                y_score = self.ap_compute_set[extra_i]['y_score'][bins_i]
                y_npos = self.ap_compute_set[extra_i]['npos'][bins_i]

                # Compute PR
                precision, recall, mark = compute_pr(y_true, y_score, y_npos)
                if not mark:
                    ap = 0
                else:
                    ap = compute_ap(precision, recall)
                if not np.isnan(ap):
                    extra_ap[extra_i][bins_i] = [ap, y_npos]
                else:
                    extra_ap[extra_i][bins_i] = [-1.0, y_npos]

        extra_ap_json = os.path.join(
            self.out_dir,
            f'epo_{self.epoch}_mAP_extra.json')
        dump_json_object(extra_ap, extra_ap_json)

        print(f'APs have been saved to {self.out_dir}')
        return {"extra_AP": extra_ap}

    def eval_hoi(self, hoi_id, global_ids, gt_dets, pred_anno,
                 mode='default', obj_cate=None):
        # flag = False
        # if hoi_id in self.verbose_list:
        #     # print(f'Evaluating hoi_id: {hoi_id} ...')
        #     flag = True

        y_true = []
        y_score = []
        det_id = []
        npos = 0
        # remaining_det = {}
        for global_id in global_ids:
            # g_flag = False
            # if global_id == "HICO_test2015_00000001" and flag:
            #     g_flag = True

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

                # if g_flag and is_match:
                #     print(sorted_idx)
                #     print(hoi_id)
                #     print(pred_det['score'])
                #     print((global_id, i))
                #     print(is_match)

            # if len(candidate_gt_dets):
            #     remaining_det[global_id] = candidate_gt_dets

            # if g_flag:
            #     print("candidate_gt_dets", candidate_gt_dets, npos)
            #     print(gt_dets[global_id])
                # print(len(sorted_idx))
                # print(f"y_ture {y_true} {len(y_true)}")
                # print(f"y_score {y_score} {len(y_score)}")

        # if len(remaining_det):
        #     print(f"hoi_id: {hoi_id}", len(remaining_det))

        # if len(remaining_det):
        #     with open("./assets/remain.json", 'a') as file:
        #         json.dump({
        #             "hoi_id": hoi_id,
        #             "remain": remaining_det
        #         }, file)
        #         file.write('\n')

        # Compute PR
        precision, recall, mark = compute_pr(y_true, y_score, npos)
        if not mark:
            ap = 0
        else:
            ap = compute_ap(precision, recall)
        # if flag:
        #     print(ap)
        # Compute AP
        # print(f'AP:{ap}')
        return ap, hoi_id

    def eval_extra(self, hoi_id, global_ids, gt_dets, pred_anno):
        npos_all = 0
        npos_extra_all = {k: 0 for k in self.extra_keys}
        for global_id in global_ids:
            img_h, img_w = self.img_size_info[global_id]
            if hoi_id in gt_dets[global_id]:
                candidate_gt_dets = gt_dets[global_id][hoi_id]
            else:
                candidate_gt_dets = []
            npos_all += len(candidate_gt_dets)

            if global_id not in pred_anno or hoi_id not in pred_anno[global_id]:
                hoi_dets = np.empty([0, 9])
            else:
                hoi_dets = pred_anno[global_id][hoi_id]

            num_dets = hoi_dets.shape[0]

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
                is_match, candidate_gt_dets, extra_info = self.match_hoi_extra(pred_det, candidate_gt_dets, img_h,
                                                                               img_w)
                for extra_i in self.extra_keys:
                    if is_match:
                        in_bins = np.min(np.argsort(np.abs(self.bins - extra_info[extra_i + '_gt']))[:2])
                        self.ap_compute_set[extra_i]['npos'][in_bins] += 1
                        npos_extra_all[extra_i] += 1
                    else:
                        in_bins = np.min(np.argsort(np.abs(self.bins - extra_info[extra_i + '_pred']))[:2])
                    self.ap_compute_set[extra_i]['y_true'][in_bins].append(is_match)
                    self.ap_compute_set[extra_i]['y_score'][in_bins].append(pred_det['score'])

            # add npos remain
            for remain_gt_det in candidate_gt_dets:
                for extra_i in self.extra_keys:
                    extra_gt = self.compute_extra[extra_i](remain_gt_det['human_box'], remain_gt_det['object_box'],
                                                           img_h, img_w)
                    in_bins = np.min(np.argsort(np.abs(self.bins - extra_gt))[:2])
                    self.ap_compute_set[extra_i]['npos'][in_bins] += 1
                    npos_extra_all[extra_i] += 1

        for extra_i in self.extra_keys:
            assert npos_extra_all[extra_i] == npos_all

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

# if __name__ == "__main__":
#     import torch
#
#     preds = torch.load("../preds.pt")
#     gts = torch.load("../gts.pt")
#     evaluator = HICOHoiHeadEvaluator(preds, gts, "../data/hico_20160224_det/", "../", -1)
#     evaluator.evaluation_extra()
