python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --use_env \
    main_fag.py \
    --pretrained pretrained/HICO_GEN_VLKT_S.pth \
    --dataset_file hico \
    --hoi_path ../gen-vlkt/data/hico_20160224_det \
    --num_obj_classes 80 \
    --num_verb_classes 117 \
    --backbone resnet50 \
    --num_queries 100 \
    --dec_layers 3 \
    --eval \
    --with_clip_label \
    --with_obj_clip_label \
    --use_nms_filter \
    --thres_nms 0.7 \
    --mixup \
    --with_sentence_branch \
