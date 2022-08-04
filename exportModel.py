'''
从权重中导出YOLOV4模型
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
import os
from customerConfig import YOLOV4Config as sys_config
if not sys_config.ISTINY:
    from nets.yolo4 import yolo_body
else:
    from nets.yolo4_tiny import yolo_body


def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
        anchors_path = os.path.expanduser(anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

def ExportYOLOModel(type,export_name, **kwargs):
    if type.upper() == 'YOLOV4':
        anchors = get_anchors(kwargs['anchors_path'])
        class_names = get_class(kwargs['classes_path'])
        model_path = os.path.expanduser(kwargs['model_path'])
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        num_anchors = len(anchors)
        num_classes = len(class_names)
        if not kwargs['istiny']:
            yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes, phi=sys_config.ATTENTION)
        else:
            yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes, phi=sys_config.ATTENTION)
        yolo_model.load_weights(model_path)
        yolo_model.compile()
        yolo_model.save(export_name, save_format='tf')
    if type.upper() == 'YOLOX':
        pass

if __name__ == '__main__':
    export_name = './village_model'
    ExportYOLOModel(type='yolov4', export_name=export_name, anchors_path=sys_config.anchors_path,
        classes_path=sys_config.classes_path,istiny=sys_config.ISTINY,model_path=sys_config.model_path)
    model = tf.keras.models.load_model(export_name)
    print('finish export model.')