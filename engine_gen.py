import copy
import itertools
import json
import math
import sys
import time
from argparse import Namespace
from typing import Iterable, Union

import numpy as np
import torch

import util.misc as utils
from datasets.hico_eval_triplet import HICOEvaluator
from datasets.sentence_eval import SentenceEvaluator
from models.gen_set_criterion import PostProcessHOITriplet
from models.hoitr import HoiTR
from models.sentence_critreion import SentenceCriterion
from models.gen_set_criterion import PostProcessHOIFag
# from datasets.vcoco_eval import VCOCOEvaluator
from datasets.hico_fag_eval import HICOFagEvaluator


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    args: Namespace,
                    epoch: int,
                    sen_criterion: SentenceCriterion = None,
                    max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # elif hasattr(criterion, 'loss_hoi_labels'):
    #     metric_logger.add_meter('hoi_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # elif hasattr(criterion, 'loss_verbs_labels'):
    #     metric_logger.add_meter('verb_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in targets]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        if args.with_sentence_branch:
            sen_loss = sen_criterion.batch_l1_loss(outputs, targets)
            loss_dict['loss_sentence_l1'] = sen_loss['l1_loss']

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        # print(loss_value)
        # sys.exit()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        # elif hasattr(criterion, 'loss_hoi_labels'):
        #     metric_logger.update(hoi_class_error=loss_dict_reduced['hoi_class_error'])
        else:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if args.dev and i >= 20:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_hoi(dataset_file,
                 model: HoiTR,
                 postprocessors: PostProcessHOITriplet,
                 data_loader,
                 subject_category_id,
                 device,
                 args,
                 sen_criterion: SentenceCriterion = None) -> dict:
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    sen_pred = []

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = samples.to(device)
        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = postprocessors['hoi'](outputs, orig_target_sizes)
        results = postprocessors(outputs, orig_target_sizes)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

        # if args.with_sentence_branch:
        #     sen_pred.extend(sen_criterion.inference(outputs))

        if args.dev and i >= 10:
            break

    if args.dev:
        return {}

    if args.with_sentence_branch:
        with open(f"{args.output_dir}/sen_result.jsonl", 'w') as file:
            for _, r in enumerate(sen_pred):
                result = json.loads(str(r).replace("\'", "\""))
                json.dump(result, file)
                file.write('\n')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    evaluator = None
    if dataset_file == 'hico':
        evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args=args)
    # elif dataset_file == 'vcoco':
    #     evaluator = VCOCOEvaluator(preds, gts, data_loader.dataset.correct_mat, use_nms_filter=args.use_nms_filter)
    stats = evaluator.evaluate()

    # if args.with_sentence_branch:
    #     sen_eval = SentenceEvaluator(sen_pred, gts, data_loader.dataset.rare_triplets,
    #                                  data_loader.dataset.non_rare_triplets, args=args)
    #     sen_eval.evaluate()

    return stats


@torch.no_grad()
def evaluate_hoi_fag(dataset_file,
                     model: HoiTR,
                     postprocessors: PostProcessHOIFag,
                     data_loader,
                     subject_category_id,
                     device,
                     args,
                     sen_criterion: SentenceCriterion = None) -> dict:
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = postprocessors['hoi'](outputs, orig_target_sizes)
        results = postprocessors(outputs, orig_target_sizes)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

        if args.dev and len(preds) >= 200:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    stats = {}
    if dataset_file == 'hico':
        evaluator = HICOFagEvaluator(preds, gts, args.hoi_path, args.output_dir,
                                     0, use_nms=args.use_nms_filter, nms_thresh=args.thres_nms)

        stats = evaluator.evaluation_default()
        print(f"(Default) mAP: {stats['mAP_def']}"
              f"mAP rare: {stats['mAP_def_rare']}"
              f"mAP non-rare: {stats['mAP_def_non_rare']}")
        stats_ko = evaluator.evaluation_ko()

        print(f"(Known Object) mAP: {stats_ko['mAP_ko']}"
              f"mAP rare: {stats_ko['mAP_ko_rare']}"
              f"mAP non-rare: {stats_ko['mAP_ko_non_rare']}")

        stats.update(stats_ko)
        # if args.eval_extra:
        #     evaluator.evaluation_extra()
    return stats
