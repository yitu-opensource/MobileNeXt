#! /bin/bash

WIDTH=$1
IMG_SIZE=$2

CUDA_VISIBLE_DEVICES=2 python imagenet.py \
        -a mobilenetv2 \
        -b 256 \
        -d ~/WSNet_improvement/imagenet/ILSVRC/Data/CLS-LOC/ \
        --epochs 200 \
        --lr-decay cos \
        --lr 0.05 \
        --wd 4e-5 \
        -c checkpoint/debug \
        --width-mult $WIDTH \
        --input-size 224 \
        -j 16 \
        -p 100 \
        --input-size $IMG_SIZE \
        #--evaluate --pretrained --weight ./pretrained/mobilenetv2_0.75-dace9791.pth \
        #--resume checkpoint/i2rnet-trans-200epoch/checkpoint.pth.tar
