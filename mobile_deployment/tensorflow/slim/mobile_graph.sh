python /home/e0357894/anaconda3/lib/python3.7/site-packages/tensorflow/python/tools/freeze_graph.py \
    --input_graph=./graph_output/i2rnet_ori_v3.pb \
    --input_checkpoint=./output/model.ckpt-100 \
    --input_binary=true --output_graph=./graph_output/frozen_i2rnet_ori_v3.pb \
    --output_node_names=MobilenetV2/Predictions/Reshape_1
