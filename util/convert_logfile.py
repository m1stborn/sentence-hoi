import json
import os

import pandas as pd

# columns = ['epoch', 'train_lr',
#            'test_mAP_def', 'test_mAP_def_rare', 'test_mAP_def_non_rare',
#            'test_mAP_ko', 'test_mAP_ko_rare', 'test_mAP_ko_non_rare',
#            'train_loss', 'train_obj_class_error', 'train_loss_sentence_l1',
#            # 'train_loss_triplet',
#            'train_loss_hoi_labels', 'train_loss_obj_ce', 'train_loss_sub_bbox',
#            'train_loss_obj_bbox', 'train_loss_sub_giou', 'train_loss_obj_giou',
#            'train_loss_hoi_labels_0', 'train_loss_obj_ce_0',
#            'train_loss_sub_bbox_0', 'train_loss_obj_bbox_0',
#            'train_loss_sub_giou_0', 'train_loss_obj_giou_0',
#            'train_loss_hoi_labels_1', 'train_loss_obj_ce_1',
#            'train_loss_sub_bbox_1', 'train_loss_obj_bbox_1',
#            'train_loss_sub_giou_1', 'train_loss_obj_giou_1',
#            'train_loss_hoi_labels_unscaled',
#            'train_hoi_class_error_unscaled', 'train_loss_obj_ce_unscaled',
#            'train_obj_class_error_unscaled', 'train_loss_sub_bbox_unscaled',
#            'train_loss_obj_bbox_unscaled', 'train_loss_sub_giou_unscaled',
#            'train_loss_obj_giou_unscaled', 'train_obj_cardinality_error_unscaled',
#            'train_loss_hoi_labels_0_unscaled', 'train_hoi_class_error_0_unscaled',
#            'train_loss_obj_ce_0_unscaled', 'train_loss_sub_bbox_0_unscaled',
#            'train_loss_obj_bbox_0_unscaled', 'train_loss_sub_giou_0_unscaled',
#            'train_loss_obj_giou_0_unscaled',
#            'train_obj_cardinality_error_0_unscaled',
#            'train_loss_hoi_labels_1_unscaled', 'train_hoi_class_error_1_unscaled',
#            'train_loss_obj_ce_1_unscaled', 'train_loss_sub_bbox_1_unscaled',
#            'train_loss_obj_bbox_1_unscaled', 'train_loss_sub_giou_1_unscaled',
#            'train_loss_obj_giou_1_unscaled',
#            'train_obj_cardinality_error_1_unscaled',
#            'train_loss_sentence_l1_unscaled',  'n_parameters']


columns = ['epoch', 'train_lr', 'test_mAP_def', 'test_mAP_def_rare', 'test_mAP_def_non_rare',
           'train_obj_class_error', 'train_loss',
           'train_loss_hoi_labels', 'train_loss_obj_ce', 'train_loss_sub_bbox',
           'train_loss_obj_bbox', 'train_loss_sub_giou', 'train_loss_obj_giou',
           'train_loss_hoi_labels_0', 'train_loss_obj_ce_0',
           'train_loss_sub_bbox_0', 'train_loss_obj_bbox_0',
           'train_loss_sub_giou_0', 'train_loss_obj_giou_0',
           'train_loss_hoi_labels_1', 'train_loss_obj_ce_1',
           'train_loss_sub_bbox_1', 'train_loss_obj_bbox_1',
           'train_loss_sub_giou_1', 'train_loss_obj_giou_1',
           'train_loss_sentence_l1', 'train_loss_hoi_labels_unscaled',
           'train_hoi_class_error_unscaled', 'train_loss_obj_ce_unscaled',
           'train_obj_class_error_unscaled', 'train_loss_sub_bbox_unscaled',
           'train_loss_obj_bbox_unscaled', 'train_loss_sub_giou_unscaled',
           'train_loss_obj_giou_unscaled', 'train_obj_cardinality_error_unscaled',
           'train_loss_hoi_labels_0_unscaled', 'train_hoi_class_error_0_unscaled',
           'train_loss_obj_ce_0_unscaled', 'train_loss_sub_bbox_0_unscaled',
           'train_loss_obj_bbox_0_unscaled', 'train_loss_sub_giou_0_unscaled',
           'train_loss_obj_giou_0_unscaled',
           'train_obj_cardinality_error_0_unscaled',
           'train_loss_hoi_labels_1_unscaled', 'train_hoi_class_error_1_unscaled',
           'train_loss_obj_ce_1_unscaled', 'train_loss_sub_bbox_1_unscaled',
           'train_loss_obj_bbox_1_unscaled', 'train_loss_sub_giou_1_unscaled',
           'train_loss_obj_giou_1_unscaled',
           'train_obj_cardinality_error_1_unscaled',
           'train_loss_sentence_l1_unscaled', 'n_parameters']


def main():
    ckpt_dir = "./checkpoint"
    for file_dir in os.listdir(ckpt_dir):
        if os.path.isdir(os.path.join(ckpt_dir, file_dir)) and file_dir != "best":
            data = []
            with open(os.path.join(ckpt_dir, file_dir, "log.txt"), encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))

            csv_filename = os.path.join(ckpt_dir, file_dir, "readable_log.csv")
            if os.path.exists(csv_filename):
                df = pd.read_csv(csv_filename)
                if len(data) <= len(df):
                    continue
            try:
                print(f"Update {csv_filename}.")
                df = pd.DataFrame(data)
                # print(df.columns)
                df = df[columns]  # reorder columns
                # print(len(df))
                df.to_csv(csv_filename, index=False)
            except KeyError:
                pass
            # break
    # print(data)


if __name__ == '__main__':
    main()
