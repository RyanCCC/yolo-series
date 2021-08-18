import onnxruntime
from PIL import Image
import numpy as np

onnx_model = './model_data/yolo_onnx.onnx'

img = './villages/JPEGImages/20210817115750.jpg'
image = Image.open(img)
image = image.convert('RGB')
boxed_image = image.resize((416,416), Image.BICUBIC)
image_data = np.array(boxed_image, dtype='float32')
image_data /= 255.
image_data = np.expand_dims(image_data, 0)


sess = onnxruntime.InferenceSession(onnx_model)
x = image_data if isinstance(image_data, list) else [image_data]
feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
pred_onnx = sess.run(None, feed)[0]
pred_onnx = np.squeeze(pred_onnx)
print('finish')