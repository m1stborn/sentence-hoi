#python dev_dataset.py \
#        --pretrained pretrained/HICO_GEN_VLKT_S.pth \
#        --dataset_file hico \
#        --hoi_path ../gen-vlkt/data/hico_20160224_det \
#        --num_obj_classes 80 \
#        --num_verb_classes 117 \
#        --backbone resnet50 \
#        --num_queries 100 \
#        --dec_layers 3 \
#        --resume checkpoint/p_202301261744/checkpoint_best.pth \
#        --eval \
#        --with_clip_label \
#        --with_obj_clip_label \
#        --with_sentence_branch \
#        --use_nms_filter \
#        --dev

python main.py \
        --pretrained pretrained/HICO_GEN_VLKT_S.pth \
        --dataset_file hico \
        --hoi_path ../gen-vlkt/data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 100 \
        --dec_layers 3 \
        --resume checkpoint/p_202301261744/checkpoint_best.pth \
        --eval \
        --with_clip_label \
        --with_obj_clip_label \
        --with_sentence_branch \
        --use_nms_filter \
        --dev


# TODO: make special sentence tensor 601 class
# TODO: add to loss (done)
# TODO: calculate sentence metric
# TODO: change "A photo of unseen interaction"
# TODO: transfer between hoi triplet\hoi text\hoi id
