#python main_gen.py \
#        --pretrained pretrained/HICO_GEN_VLKT_S.pth \
#        --dataset_file hico \
#        --hoi_path ../gen-vlkt/data/hico_20160224_det \
#        --num_obj_classes 80 \
#        --num_verb_classes 117 \
#        --backbone resnet50 \
#        --num_queries 64 \
#        --dec_layers 3 \
#        --eval \
#        --with_clip_label \
#        --with_obj_clip_label \
#        --use_nms_filter \
#        --dev

# checkpoint p_202301131633
#    epoch 20 - current best

# checkpoint p_202301150236
#   epoch 30 "test_mAP": 0.16325010839973483, "test_mAP rare": 0.07872934083694318, "test_mAP non-rare": 0.18951903040987958
#   epoch 32 "test_mAP": 0.15419893278586058, "test_mAP rare": 0.09117042572750175, "test_mAP non-rare": 0.17381498403469936

# checkpoint p_202301160140
#   epoch 40 "test_mAP": 0.1764803197899045
#   epoch 78 "test_mAP": 0.17191885609839438

#python main_gen.py \
#        --pretrained pretrained/HICO_GEN_VLKT_S.pth \
#        --dataset_file hico \
#        --hoi_path ../gen-vlkt/data/hico_20160224_det \
#        --num_obj_classes 80 \
#        --num_verb_classes 117 \
#        --backbone resnet50 \
#        --num_queries 64 \
#        --dec_layers 3 \
#        --resume checkpoint/p_202301160140/checkpoint_best.pth \
#        --eval \
#        --with_clip_label \
#        --with_obj_clip_label \
#        --use_nms_filter \
#        >> "./checkpoint/tmp_stdout.txt"
#        --dev \

# Num Query 100
#python main_gen.py \
#        --pretrained pretrained/HICO_GEN_VLKT_S.pth \
#        --dataset_file hico \
#        --hoi_path ../gen-vlkt/data/hico_20160224_det \
#        --num_obj_classes 80 \
#        --num_verb_classes 117 \
#        --backbone resnet50 \
#        --num_queries 100 \
#        --dec_layers 3 \
#        --resume checkpoint/p_202301250128/checkpoint_best.pth \
#        --eval \
#        --with_clip_label \
#        --with_obj_clip_label \
#        --use_nms_filter \
#        >> "./checkpoint/tmp_stdout.txt"
#        --dev \

# checkpoint p_202301220301
#   epoch 14 "test_mAP": 0.16118728981241318

# checkpoint p_202301250128
# epoch 52 "test_mAP": 0.20934590381517001 -> best

# checkpoint p_202301260418
# epoch 54 "test_mAP": 0.2092760241297853

# Feat: hoitr and gen-vlkt version 1 - ckptp - 202301260418

#######################################################

python main.py \
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
#        >> "./checkpoint/tmp_stdout.txt"
#        --dev \

# checkpoint p_202301261744
# epoch 70 "test_mAP": 0.24306954216542398

# move tmep logfile to folder create by main_gen.py
mv "./checkpoint/tmp_stdout.txt" "$(ls -td ./checkpoint/*/ | head -1)stdout.txt"
