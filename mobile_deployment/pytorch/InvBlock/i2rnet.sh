#! /bin/bash

WIDTH=$1

CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet.py \
        -a i2rnetv3 \
        -b 512 \
        -d ~/WSNet_improvement/imagenet/ILSVRC/Data/CLS-LOC/ \
        --epochs 200 \
        --lr-decay cos \
        --lr 0.1 \
        --wd 4e-5 \
        -c checkpoint/i2rnet_low_id_mbv2 \
        --width-mult $WIDTH \
        --input-size 224 \
        -j 64 \
        -p 100 \
        #--resume checkpoint/i2rnet-trans-200epoch/checkpoint.pth.tar
