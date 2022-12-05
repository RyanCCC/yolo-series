import tensorflow as tf

saved_model_dir = './model/yolox_model'
converter  = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()

# 保存tflite模型

print('finish')
