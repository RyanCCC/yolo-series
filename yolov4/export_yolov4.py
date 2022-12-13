from cfg import YOLOV4Config
from .nets.yolo4 import yolo_body
import os
from tensorflow.keras.layers import Input
import tensorflow as tf
import tf2onnx
import numpy as np

def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names



def export_model(weights, saved_pb, saved_pb_dir, opset, onnx_save_path):
    anchors = get_anchors(YOLOV4Config.anchors_path)
    class_names = get_class(YOLOV4Config.classes_path)
    model_path = os.path.expanduser(weights)
    assert model_path.endswith('.h5'), 'Tensorflow model or weights must be a .h5 file.'
    num_anchors = len(anchors)
    num_classes = len(class_names)
    yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes, phi=YOLOV4Config.ATTENTION)
    yolo_model.load_weights(model_path)
    yolo_model.compile()
    if saved_pb:
        assert len(saved_pb_dir) > 0, 'save_name cannot be none or empty.'
        yolo_model.save(saved_pb_dir, save_format='tf')
    model_proto, _ = tf2onnx.convert.from_keras(yolo_model, opset=opset, output_path=onnx_save_path)
    output_names = [n.name for n in model_proto.graph.output]
    print(f'Model output names: ',output_names)