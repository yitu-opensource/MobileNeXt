# create TF-Records for houseplants dataset

python create_tfrecord.py --dataset_dir=../../../hp_dataset --tfrecord_filename=hp_plants --validation_size=0.2


# Variables 
DATASET_DIR=../../../hp_dataset
TRAIN_DIR=./train_dir
CHECKPOINT_PATH=./mobilenet_v1_1.0_224.ckpt

# Training on CPU
python train_image_classifier.py \
    --train_dir=./train_dir \
    --dataset_dir=../../../hp_dataset \
    --dataset_name=hp_plants \
    --dataset_split_name=train \
    --model_name=mobilenet_v1 \
    --train_image_size=224 \
    --checkpoint_path=./mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt \
    --max_number_of_steps=30000 \
    --clone_on_cpu=True \
    --checkpoint_exclude_scopes=MobilenetV1/Logits


# Training on GPU
python train_image_classifier.py \
    --train_dir=./train_dir \
    --dataset_dir=../../../hp_dataset\
    --dataset_name=hp_plants \
    --dataset_split_name=train \
    --model_name=mobilenet_v1 \
    --train_image_size=224 \
    --checkpoint_path=./mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt \
    --max_number_of_steps=30000 \
    --checkpoint_exclude_scopes=MobilenetV1/Logits

# Evaluation

python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=./train_dir/model.ckpt-30000 \
    --dataset_dir=../../../hp_dataset \
    --dataset_name=hp_plants \
    --dataset_split_name=validation \
    --model_name=mobilenet_v1 \
    --eval_image_size=224



# create inference graph 

python export_inference_graph.py \
  --alsologtostderr \
  --model_name=mobilenet_v1 \
  --output_file=./inference_graph_mobilenet.pb \
  --dataset_name=hp_plants

# export frozen graph 
python freeze_graph.py \
    --input_graph=./inference_graph_mobilenet.pb \
    --input_binary=true \
    --input_checkpoint=./train_dir/model.ckpt-30000 \
    --output_graph=./frozen_mobilenet.pb \
    --output_node_names=MobilenetV1/Predictions/Reshape_1

# optimize graph
python optimize_for_inference.py \
    --input=./frozen_mobilenet.pb \
    --output=./opt_frozen_mobilenet.pb \
    --input_names=input \
    --output_names=MobilenetV1/Predictions/Reshape_1



  