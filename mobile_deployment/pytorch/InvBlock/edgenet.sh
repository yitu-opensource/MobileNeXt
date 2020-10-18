#! /bin/bash

WIDTH=$1

CUDA_VISIBLE_DEVICES=4,5,6,7 python imagenet.py \
        -a edgenet \
        -d /temp/zhoudaquan/imagenet/ILSVRC/Data/CLS-LOC/ \
        --epochs 200 \
        --lr-decay cos \
        --lr 0.05 \
        --wd 4e-5 \
        -c checkpoint/edgenet-200epoch \
        --width-mult $WIDTH \
        --input-size 224 \
        -j 8 \
        -p 100 \
        #--resume checkpoint/i2rnet-trans-200epoch/checkpoint.pth.tar
