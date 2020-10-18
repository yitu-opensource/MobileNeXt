DATASET_DIR= /home/e0357894/WSNet_improvement/imagenet_tfrecord/
TRAIN_DIR=/home/e0357894/zhoudaquan/eccv20/tensorflow-train-to-mobile-tutorial/models/research/slim
python train_image_classifier.py \
    --train_dir ./output/ \
    --max_number_of_steps 100 \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --dataset_dir /home/e0357894/WSNet_improvement/imagenet_tfrecord/ \
    --model_name=mobilenet_v2
