import json
import os


def main():
    # load gt
    # Load anno_list
    print('Loading anno_list.json ...')
    anno_list = json.load(open(os.path.join("./data/annotations/anno_list.json", 'anno_list.json'), "r"))

    gt_dets = {}
    for anno in anno_list:
        if "test" not in anno['global_id']:
            continue

        global_id = anno['global_id']
        gt_dets[global_id] = {}
        img_h, img_w, _ = anno['image_size']
        # img_size_info[global_id] = [img_h, img_w]
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

    print(gt_dets)
    # load remain
    with open("./assets/remain.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            result = json.loads(line)
            print(result)
            break

    return


if __name__ == "__main__":
    main()
