#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet.py \
        -a mobilenetv2_high_dim -b 256 \
        -d ~/WSNet_improvement/imagenet/ILSVRC/Data/CLS-LOC/ \
        --epochs 200 \
        --lr-decay cos \
        --lr 0.05 \
        --wd 4e-5 \
        -c checkpoint/mobilenetv2-dual-shortcut \
        --width-mult 1. \
        --input-size 224 \
        -j 64 \
        -p 100 #| tee logs/mobilenetv2.log
