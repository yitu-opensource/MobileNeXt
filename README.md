

## Updates 

### Oct 26, 2020
* Code adapted from repo [yitu-opensource MobileNeXt](https://github.com/yitu-opensource/MobileNeXt)

### Oct 27, 2020
* Code for detection is [available](https://github.com/Andrew-Qibin/ssdlite-pytorch-mobilenext)

# rethinking_bottleneck_design
This repo contains the code for the paper Rethinking Bottleneck Structure for Efficient Mobile Network Design ([ECCV 2020](https://arxiv.org/pdf/2007.02269.pdf))

MNEXT is an light weight models cater for mobile devices. It combines the advantages of traditional ResNet bottleneck building block and the MBV2 inverted residual block. Besides, the newly proposed building block also takes the hardware implementation into consideration such that the memory consumption can be adjusted at algorithm level without minimum impacts on the model performance.

```
@article{zhou2020rethinking,
  title={Rethinking Bottleneck Structure for Efficient Mobile Network Design},
  author={Zhou, Daquan and Hou, Qibin and Chen, Yunpeng and Feng, Jiashi and Yan, Shuicheng},
  journal={ECCV, August},
  year={2020}
}
```

The training framework is modified based on an older version(upon release) of the repo  [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
## Performance
Model performance at different width multiplier:
Model|Param.(M)|Madd(M)|Top-1 Acc.(%)
---|---|---|---
MobileNeXt-1.40|6.1|590|76.1
MobileNeXt-1.00|3.5|300|74.02
MobileNeXt-0.75|2.5|210|72
MobileNeXt-0.50|2.1|110|67.7
MobileNeXt-0.35|1.8|80|64.7

Latency and accuracy with different tensor multiplier

Model|Tensor multiplier|Madd(M)|Top-1 Acc.(%)|Latency(Pytorch,ms)
---|---|---|---|---
MobileNeXt|6.1|300|74.02|211
MobileNeXt|3.5|300|74.09|196
MobileNeXt|2.5|300|73.91|195
MobileNeXt|2.1|300|73.68|188

Latency measurement with Pytorch and TF Lite:
Model|Pixel 4-CPU(ms)|Pixel 4-GPU(ms)|Platform
---|---|---|---
MBV2|190 - 220|-|Pytorch Mobile
Ours|191 - 220|-|Pytorch Mobile
MBV2|68 - 92|-|TF Mobile
Ours|66 - 91|-|TF Mobile
MBV2|44 - 47|15 - 17|TF Lite
Ours|48 - 51|16 - 17|TF Lite

## To reproduce the results in the paper
run the batch script as below:
bash mnext_efficient_l.sh # of process

The three scripts are used for the training from scratch.

1. mnext_efficient_l.sh: MobileNeXt large model based on EfficientNet backbone
2. mnext_efficient_s.sh: MobileNeXt small model based on EfficientNet backbone
3. mnext_mbv2.sh: MobileNeXt large model based on MobileNetV2 backbone

## To reproduce the latency measurement 
TF lite
There are four steps to follow:
change to mobile deployment folder:
1. Run the MobileNeXt model tf version to save a checkpoint
* bash run.sh
    ```
        DATASET_DIR= /home/e0357894/WSNet_improvement/imagenet_tfrecord/
        TRAIN_DIR=/home/e0357894/zhoudaquan/eccv20/tensorflow-train-to-mobile-tutorial/models/research/slim
        python train_image_classifier.py \
        --train_dir ./output/ \
        --max_number_of_steps 100 \
        --dataset_name=imagenet \
        --dataset_split_name=train \
        --dataset_dir /home/e0357894/WSNet_improvement/imagenet_tfrecord/ \
        --model_name=mobilenet_v2
    ```
2. convert the checkpoint to mobile graph
* bash mobile_graph.sh
    ```
        python /home/e0357894/anaconda3/lib/python3.7/site-packages/tensorflow/python/tools/freeze_graph.py \
        --input_graph=./graph_output/i2rnet_ori_v3.pb \
        --input_checkpoint=./output/model.ckpt-100 \
        --input_binary=true --output_graph=./graph_output/frozen_i2rnet_ori_v3.pb \
        --output_node_names=MobilenetV2/Predictions/Reshape_1
    ```
3. generate the freeze graph based on the mobile graph
* bash convert.sh
    ```
        python export_inference_graph.py \
            --alsologtostderr \
            --model_name=mobilenet_v2 \
            --output_file=graph_output/i2rnet_ori_v3.pb 
    ```
4. convert to tf lite model to be loaded by android studio
* python tflite_converter.py
```
    import tensorflow.lite as lite

    # Converting a SavedModel.
    saved_model_dir = "./output/model.ckpt-100"
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    open("converted_model.tflite", "wb").write(tflite_model)
```

After generating the mobile model based on tf lite, copy the model to the assets folder under the android studio project

## Code for detection on both Pascal VOC and COCO

Please refer to this [repo](https://github.com/Andrew-Qibin/ssdlite-pytorch-mobilenext). Our MobileNeXt improves MobileNetV2 by
1% (from 22.3% to 23.3%) in terms of mAP under the same settings.

## To do : 
1. Add in Android project for apk generation
