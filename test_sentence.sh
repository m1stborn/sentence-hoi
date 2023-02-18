#python main_fag.py \
#        --pretrained pretrained/HICO_GEN_VLKT_S.pth \
#        --dataset_file hico \
#        --hoi_path ../gen-vlkt/data/hico_20160224_det \
#        --num_obj_classes 80 \
#        --num_verb_classes 117 \
#        --backbone resnet50 \
#        --num_queries 100 \
#        --dec_layers 3 \
#        --eval \
#        --with_clip_label \
#        --with_obj_clip_label \
#        --use_nms_filter \
#        --use_fag_setting \
#        --resume checkpoint/p_202302062134/checkpoint_best.pth \
#        --with_sentence_branch \
#        --dev \
#        >> "./checkpoint/tmp_stdout.txt"

python test_fag.py \
        --resume checkpoint/p_202302150339/checkpoint_best.pth \
#        --pretrained pretrained/HICO_GEN_VLKT_S.pth \
#        --dataset_file hico \
#        --hoi_path ../gen-vlkt/data/hico_20160224_det \
#        --num_obj_classes 80 \
#        --num_verb_classes 117 \
#        --backbone resnet50 \
#        --num_queries 100 \
#        --dec_layers 3 \
#        --eval \
#        --with_clip_label \
#        --with_obj_clip_label \
#        --use_nms_filter \
#        --use_fag_setting \
#        --with_sentence_branch \
#        --dev

# move tmep logfile to folder create by main_gen.py
#mv "./checkpoint/tmp_stdout.txt" "$(ls -td ./checkpoint/*/ | head -1)stdout.txt"
