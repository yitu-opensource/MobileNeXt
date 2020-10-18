import tensorflow as tf
# Path to the frozen graph file
graph_def_file = 'graph_output/frozen_i2rnet_ori_v3.pb'
# A list of the names of the model's input tensors
input_arrays = ['input']
# A list of the names of the model's output tensors
output_arrays = ['MobilenetV2/Predictions/Reshape_1']
# Load and convert the frozen graph
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
          graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
# Write the converted model to disk
open("graph_output/i2rnet_ori_v2_1_0_1_dwise.tflite", "wb").write(tflite_model)
