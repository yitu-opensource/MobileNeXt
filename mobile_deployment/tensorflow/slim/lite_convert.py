import tensorflow.lite as lite

# Converting a SavedModel.
saved_model_dir = "./output/model.ckpt-100"
converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

open("converted_model.tflite", "wb").write(tflite_model)
