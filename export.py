'''
将训练Tensorflow模型导出ONNX：
1. 从权重模型导出pb模型
2. 从权重模型导出ONNX模型
3. 从pb模型导出成ONNX模型



Tensorflow 模型命名规则：数据集（功能）_算法模型_版本.h5
ONNX模型命名规则：数据集功能_算法模型_OP_输入维度_版本.onnx

Usage:
    python .\export.py --weight .\model\VOC.h5 --yolo yolox --save_onnx 'voc_yolox_l_13_640_v1.onnx' 

'''

import argparse
import os
import numpy as np


def parse_arg():
    parser = argparse.ArgumentParser(description="Export YOLO model")
    parser.add_argument('--weight', type=str, help='model weight', default='')
    parser.add_argument('--saved_pb', action='store_true', help='save pb model to current directory')
    parser.add_argument('--saved_pb_dir', type=str, default='./save_model', help='save pb file if needed. Default:save_model')
    parser.add_argument('--yolo', type=str, help='YOLO algorithm.', choices=['yolov4', 'yolov4_tiny', 'yolox'], required=True)
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
        if yolo_type == 'yolov4' or yolo_type == 'yolov4_tiny':
            from yolov4 import export_model
            save_pb = args.saved_pb
            save_name = args.saved_pb_dir
            export_model(weights, save_pb, save_name, opset=opset, onnx_save_path=onnx_save_path)
            print('success export YOLOV4.')
        elif yolo_type == 'yolox':
            from yolox import export_model
            save_pb = args.saved_pb
            save_name = args.saved_pb_dir
            export_model(weights, save_pb, save_name, opset=opset, onnx_save_path=onnx_save_path)
            print('success export YOLOX.')
if __name__ == '__main__':
    parser = parse_arg()
    args = parser.parse_args()
    main(args=args)