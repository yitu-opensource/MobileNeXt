#! /bin/bash

WIDTH=$1

CUDA_VISIBLE_DEVICES=2,3 python imagenet.py \
        -a i2rnetv2 \
        -d ~/WSNet_improvement/imagenet/ILSVRC/Data/CLS-LOC/ \
        --epochs 200 \
        --lr-decay cos \
        --lr 0.05 \
        --wd 4e-5 \
        -c checkpoint/i2rnetv2-200epoch \
        --width-mult $WIDTH \
        --input-size 224 \
        -j 16 \
        -p 100 \
        #--resume checkpoint/i2rnet-trans-200epoch/checkpoint.pth.tar
