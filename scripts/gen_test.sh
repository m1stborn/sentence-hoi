python test_gen.py \
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
        --use_nms_filter \
#        --dev
#        >> "./checkpoint/tmp_stdout.txt"

# move tmep logfile to folder create by main_gen.py
#mv "./checkpoint/tmp_stdout.txt" "$(ls -td ./checkpoint/*/ | head -1)stdout.txt"
