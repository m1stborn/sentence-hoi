python main_fag.py \
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
        --use_fag_setting \
        --dev \
        --with_sentence_branch \
#        --resume checkpoint/p_202302062134/checkpoint_best.pth \
#        >> "./checkpoint/tmp_stdout.txt"

# move tmep logfile to folder create by main_gen.py
#mv "./checkpoint/tmp_stdout.txt" "$(ls -td ./checkpoint/*/ | head -1)stdout.txt"

# checkpoint p_202302062134
# epoch 123 "test_mAP_def": 0.23368687085219425

# TODO: label composition by word2vec, then encoding by clip
# TODO: calculate sentence metric
# TODO: make special sentence tensor 601 class
# TODO: change "A photo of unseen interaction"
# TODO: transfer between hoi triplet\hoi text\hoi id (done)
# TODO: add to loss (done)