#!/bin/bash
NUM_PROC=$1
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py "$@" ~/WSNet_improvement/imagenet/ILSVRC/Data/CLS-LOC/ \
    --model mnext_mbv2_cfg --num-gpu 1 \
    --lr 0.1 --sched cosine --decay-rate 1 \
    --opt sgd --epochs 200 -b 128 --log-interval 500 -j 8 --model-ema \
    --smoothing 0.1 --output ./output/mnext_mbv2/ \
    --resume output/74_09_i2rnetv3_half_id/model_best.pth.tar
    # --eval-only \
    # --resume ./output/mbv2_100_1_3_tensor_continue/train/gal-daquanzhou-dlqf7krpor-6t7zs-20200222-084840-mbv2_100-224/model_best.pth.tar
    # --resume ./output/i2rnetv3_debug/train/gal-daquanzhou-l7fd2f8b9d-h9xtv-20200220-041740-i2rnetv3_more_1x1-224/checkpoint-106.pth.tar
    # --resume output/train/74_09_i2rnetv3_half_id/model_best.pth.tar
    # --resume output/train/74_02_full_id_tensor/model_best.pth.tar

