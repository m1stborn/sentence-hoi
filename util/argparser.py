import json
import os
import getpass
import argparse
import datetime
from pathlib import Path


def create_log_dir(checkpoint='checkpoint', log_path='/data/LOG/train_log'):
    base_dir = os.path.join(log_path, getpass.getuser())
    exp_name = os.path.basename(os.path.abspath('.'))
    log_dir = os.path.join(base_dir, exp_name)
    print(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(checkpoint):
        cmd = "ln -s {} {}".format(log_dir, checkpoint)
        os.system(cmd)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--lr_clip', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # Backbone
    parser.add_argument('--backbone', choices=['resnet50', 'resnet101', 'swin'],
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # parser.add_argument('--backbone', default='resnet50', type=str,
    #                     help="Name of the convolutional backbone to use")
    # parser.add_argument('--dilation', action='store_true',
    #                     help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    # parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
    #                     help="Type of positional embedding to use on top of the image features")

    # Transformer
    # parser.add_argument('--enc_layers', default=6, type=int,
    #                     help="Number of encoding layers in the transformer")
    # parser.add_argument('--dec_layers', default=3, type=int,
    #                     help="Number of stage1 decoding layers in the transformer")
    # parser.add_argument('--dim_feedforward', default=2048, type=int,
    #                     help="Intermediate size of the feedforward layers in the transformer blocks")
    # parser.add_argument('--hidden_dim', default=256, type=int,
    #                     help="Size of the embeddings (dimension of the transformer)")
    # parser.add_argument('--dropout', default=0.1, type=float,
    #                     help="Dropout applied in the transformer")
    # parser.add_argument('--nheads', default=8, type=int,
    #                     help="Number of attention heads inside the transformer's attentions")
    # parser.add_argument('--num_queries', default=100, type=int,
    #                     help="Number of query slots")
    # parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Segmentation
    # parser.add_argument('--masks', action='store_true',
    #                     help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--hoi', action='store_true',
                        help="Train for HOI if the flag is provided")
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # parser.add_argument('--with_mimic', action='store_true',
    #                     help="Use clip feature mimic")
    # Matcher
    # parser.add_argument('--set_cost_class', default=1, type=float,
    #                     help="Class coefficient in the matching cost")
    # parser.add_argument('--set_cost_bbox', default=5, type=float,
    #                     help="L1 box coefficient in the matching cost")
    # parser.add_argument('--set_cost_giou', default=2, type=float,
    #                     help="giou box coefficient in the matching cost")

    # Gen
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=2.5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")
    parser.add_argument('--set_cost_hoi', default=1, type=float,
                        help="Hoi class coefficient")

    # Loss coefficients
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=2.5, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.02, type=float,
                        help="Relative classification weight of the no-object class")

    # parser.add_argument('--dice_loss_coef', default=1, type=float)
    # parser.add_argument('--bbox_loss_coef', default=2.5, type=float)
    # parser.add_argument('--giou_loss_coef', default=1, type=float)
    # parser.add_argument('--mask_loss_coef', default=1, type=float)

    parser.add_argument('--obj_loss_coef', default=1, type=float)
    # parser.add_argument('--verb_loss_coef', default=2, type=float)
    parser.add_argument('--hoi_loss_coef', default=1, type=float)
    # parser.add_argument('--mimic_loss_coef', default=20, type=float)
    parser.add_argument('--alpha', default=0.5, type=float, help='focal loss alpha')
    # parser.add_argument('--eos_coef', default=0.1, type=float,
    #                     help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='hico')
    # parser.add_argument('--coco_path', type=str)
    # parser.add_argument('--coco_panoptic_path', type=str)
    # parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', type=str)
    exp_time = datetime.datetime.now().strftime('%Y_%m%d_%H%M')
    create_log_dir(checkpoint='checkpoint', log_path='/home')
    work_dir = 'checkpoint/p_{}'.format(exp_time)

    parser.add_argument('--output_dir', default=work_dir,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # hoi eval parameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.5, type=float)

    # For Fag setting
    # parser.add_argument('--thres_nms', default=0.5, type=float)
    parser.add_argument('--nms_alpha', default=1, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)
    parser.add_argument('--json_file', default='results.json', type=str)

    # clip
    parser.add_argument('--ft_clip_with_small_lr', action='store_true',
                        help='Use smaller learning rate to finetune clip weights')
    parser.add_argument('--with_clip_label', action='store_true', help='Use clip to classify HOI')
    # parser.add_argument('--early_stop_mimic', action='store_true', help='stop mimic after step')
    parser.add_argument('--with_obj_clip_label', action='store_true', help='Use clip to classify object')
    # parser.add_argument('--clip_model', default='ViT-B/32', help='clip pretrained model path')
    # parser.add_argument('--fix_clip', action='store_true', help='')
    parser.add_argument('--clip_embed_dim', default=512, type=int)

    # zero shot type
    parser.add_argument('--zero_shot_type', default='default',
                        help='default, rare_first, non_rare_first, unseen_object, unseen_verb')
    parser.add_argument('--del_unseen', action='store_true', help='')

    parser.add_argument('--dev', action='store_true', help='')

    # sentence hoi
    parser.add_argument('--with_sentence_branch', action='store_true', help='Use sentence branch')
    parser.add_argument('--sentence_l1_loss_coef', default=0.3, type=float)
    parser.add_argument('--sentence_triplet_loss_coef', default=5, type=float)

    parser.add_argument('--use_fag_setting', action='store_true', help='Use Fag setting for evaluator, postprocessor..')
    parser.add_argument('--no_obj', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--eval_bbox', action='store_true')

    return parser


def save_args(arg):
    with open(os.path.join(arg.output_dir, "args.json"), 'wt') as f:
        json.dump(vars(arg), f, indent=4)


def load_args(filepath):
    with open(os.path.join(filepath, "args.json"), 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        arg = parser.parse_args(namespace=t_args)

    return args


if __name__ == '__main__':

    parser = argparse.ArgumentParser('SentenceHOI training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
