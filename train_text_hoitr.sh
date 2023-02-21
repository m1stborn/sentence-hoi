# Interaction Head: HOI
python main_text_hoitr.py \
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
        --with_sentence_branch \
        --thres_nms 0.7 \
        --no_aux_loss