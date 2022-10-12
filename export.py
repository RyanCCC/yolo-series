'''
将训练模型导出ONNX：
1. 从权重模型导出pb模型
2. 从权重模型导出ONNX模型
3. 从pb模型导出成ONNX模型

Tensorflow 模型命名规则：数据集（功能）_算法模型_版本.h5
ONNX模型命名规则：数据集功能_算法模型_OP_输入维度_版本.onnx
'''

import argparse
import os
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

def parse_arg():
    parser = argparse.ArgumentParser(description="Export YOLO model")
    parser.add_argument('--weight', type=str, help='model weight', default='')
    parser.add_argument('--saved_pb', action='store_true', help='save pb model to current directory')
    parser.add_argument('--saved_pb_dir', type=str, default='./save_model', help='save pb file if needed. Default:save_model')
    parser.add_argument('--yolo', type=str, help='YOLO algorithm.', choices=['yolov4', 'yolov4_tiny','yolov5', 'yolox'], required=True)
    parser.add_argument('--saved_model', type=str, help='Tensorflow saved_model', default='')
    parser.add_argument('--save_onnx', type=str, help='save onnx model name', required=True, default='')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--flag', action='store_true', help='True:Tensoflow model, False:Tensorflow weights')
    return parser

def main(args):
    from tensorflow.keras.layers import Input
    import tensorflow as tf
    import tf2onnx
    onnx_save_path = args.save_onnx
    opset = args.opset
    if args.flag:
        '''
        从tensorflow模型中导出onnx模型
        '''
        saved_model = args.saved_model
        assert len(saved_model) > 0, 'saved_model cannot be none or empty.'
        yolo_model = tf.keras.models.load_model(saved_model)
        model_proto, _ = tf2onnx.convert.from_keras(yolo_model, opset=opset, output_path=onnx_save_path)
        output_names = [n.name for n in model_proto.graph.output]
        print(output_names)
    else:
        '''
        从权重中导出模型
        '''
        yolo_type = args.yolo
        print('Convert Tensorflow saved model to ONNX')
        weights = args.weight
        assert len(weights) > 0, 'weights cannot be none or empty.'
        if yolo_type == 'yolov4':
            from customerConf import YOLOV4Config
            from yolov4.nets.yolo4 import yolo_body
            anchors = get_anchors(YOLOV4Config.anchors_path)
            class_names = get_class(YOLOV4Config.classes_path)
            model_path = os.path.expanduser(weights)
            assert model_path.endswith('.h5'), 'Tensorflow model or weights must be a .h5 file.'
            num_anchors = len(anchors)
            num_classes = len(class_names)
            yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes, phi=YOLOV4Config.ATTENTION)
            yolo_model.load_weights(model_path)
            yolo_model.compile()
            save_pb = args.saved_pb
            if save_pb:
                save_name = args.saved_pb_dir
                assert len(save_name) > 0, 'save_name cannot be none or empty.'
                yolo_model.save(save_name, save_format='tf')
            model_proto, _ = tf2onnx.convert.from_keras(yolo_model, opset=opset, output_path=onnx_save_path)
            output_names = [n.name for n in model_proto.graph.output]
            print(f'Model output names: ',output_names)
        elif yolo_type == 'yolov4_tiny':
            from customerConf import YOLOV4Config
            from yolov4.nets.yolo4_tiny import yolo_body
            anchors = get_anchors(YOLOV4Config.anchors_path)
            class_names = get_class(YOLOV4Config.classes_path)
            weight_path = os.path.expanduser(weights)
            assert weight_path.endswith('.h5'), 'Tensorflow model or weights must be a .h5 file.'
            num_anchors = len(anchors)
            num_classes = len(class_names)
            yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes, phi=YOLOV4Config.ATTENTION)
            yolo_model.load_weights(weight_path)
            yolo_model.compile()
            save_pb = args.saved_pb
            if save_pb:
                save_name = args.saved_pb_dir
                assert len(save_name) > 0, 'save_name cannot be none or empty.'
                yolo_model.save(save_name, save_format='tf')
            model_proto, _ = tf2onnx.convert.from_keras(yolo_model, opset=opset, output_path=onnx_save_path)
            output_names = [n.name for n in model_proto.graph.output]
            print(f'Model output names: ',output_names)
        elif yolo_type == 'yolov5':
            # TODO
            pass
        elif yolo_type == 'yolox':
            from customerConf import YOLOXConfig
            from yolox.nets.yolox import yolo_body
            class_names = get_class(YOLOXConfig.classes_path)
            num_classes = len(class_names)
            yolo_model = yolo_body([None, None, 3], num_classes=num_classes, phi=YOLOXConfig.phi)
            weight_path = os.path.expanduser(weights)
            assert weight_path.endswith('.h5'), 'Tensorflow model or weights must be a .h5 file.'
            yolo_model.load_weights(weight_path)
            print('model weight success load.')
            if save_pb:
                save_name = args.saved_pb_dir
                assert len(save_name) > 0, 'save_name cannot be none or empty.'
                yolo_model.save(save_name, save_format='tf')
            model_proto, _ = tf2onnx.convert.from_keras(yolo_model, opset=opset, output_path=onnx_save_path)
            output_names = [n.name for n in model_proto.graph.output]
            print(f'Model output names: ',output_names)
        elif yolo_type == 'yolov7':
            # TODO
            pass
if __name__ == '__main__':
    parser = parse_arg()
    args = parser.parse_args()
    main(args=args)