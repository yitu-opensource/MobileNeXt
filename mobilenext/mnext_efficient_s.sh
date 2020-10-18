#!/bin/bash
NUM_PROC=$1
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py "$@" ~/WSNet_improvement/imagenet/ILSVRC/Data/CLS-LOC/ \
    --model mnext_s --num-gpu 1 --lr 0.12 --sched step --decay-rate 0.97  \
    --opt rmsproptf --opt-eps 0.001 --warmup-lr 1e-6 \
    --decay-epochs 2.4 --epochs 850 \
    -b 128 --log-interval 550 -j 64 --smoothing 0.1 \
    --weight-decay 1e-5 --model-ema-decay 0.9999 --model-ema \
    --output ./output/efficienti2rnet_b1_5_68_amp_ra_lr_012_continue --log-interval 500 \
    --drop 0.2 --drop-connect 0.2 --reprob 0.2 --amp \
    --resume ./output/76_05_3_94_model/model_best.pth.tar\
    # --resume ./output/4_29M_76_6/model_best.pth.tar\
    # --remode pixel --grid \
    # --aa rand-m9-mstd0.5 --remode pixel --grid \

