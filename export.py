'''
将训练Tensorflow模型导出ONNX：
1. 从权重模型导出pb模型
2. 从权重模型导出ONNX模型
3. 从pb模型导出成ONNX模型



Tensorflow 模型命名规则：数据集（功能）_算法模型_版本.h5
ONNX模型命名规则：数据集功能_算法模型_OP_输入维度_版本.onnx

Usage:
    python .\export.py --model .\model\VOC.h5 --yolo yolox --save_onnx 'voc_yolox_l_13_640_v1.onnx' 

'''

import argparse
import os
import numpy as np


def parse_arg():
    parser = argparse.ArgumentParser(description="Export YOLO model")
    parser.add_argument('--model', type=str, help='yolo model', default='')
    parser.add_argument('--saved_pb', action='store_true', help='save pb model to current directory')
    parser.add_argument('--saved_pb_dir', type=str, default='./save_model', help='save pb file if needed. Default:save_model')
    parser.add_argument('--yolo', type=str, help='YOLO algorithm.', choices=['yolov4', 'yolov4_tiny', 'yolov5', 'yolov5-v61', 'yolox', 'yolov7'], required=True)
    parser.add_argument('--saved_model', type=str, help='Tensorflow saved_model', default='')
    parser.add_argument('--save_onnx', type=str, help='save onnx model name', required=True, default='')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    return parser

def main(args):
    from tensorflow.keras.layers import Input
    import tensorflow as tf
    import tf2onnx
    onnx_save_path = args.save_onnx
    opset = args.opset
    yolo_type = args.yolo
    model = args.model
    assert len(model) > 0, 'weights cannot be none or empty.'
    if yolo_type == 'yolov4' or yolo_type == 'yolov4_tiny':
        from yolov4 import export_yolov4
        save_pb = args.saved_pb
        save_name = args.saved_pb_dir
        export_yolov4(model, save_pb, save_name, opset=opset, onnx_save_path=onnx_save_path)
        print('success export YOLOV4.')
    elif yolo_type == 'yolox':
        from yolox import export_yolox
        save_pb = args.saved_pb
        save_name = args.saved_pb_dir
        export_yolox(model, save_pb, save_name, opset=opset, onnx_save_path=onnx_save_path)
        print('success export YOLOX.')
    elif yolo_type == 'yolov5':
        from yolov5 import export_yolov5
        save_pb = args.saved_pb
        save_name = args.saved_pb_dir
        export_yolov5(model, save_pb, save_name, opset=opset, onnx_save_path=onnx_save_path)
        print('success export yolov5.')
    elif yolo_type == 'yolov5-v61':
        from yolov5v61 import export_yolov5v61
        save_pb = args.saved_pb
        save_name = args.saved_pb_dir
        export_yolov5v61(model, save_pb, save_name, opset=opset, onnx_save_path=onnx_save_path)
        print('success export yolov5-v61.')
    elif yolo_type == 'yolov7':
        from yolov7 import export_yolov7
        save_pb = args.saved_pb
        save_name = args.saved_pb_dir
        export_yolov7(model, save_pb, save_name, opset=opset, onnx_save_path=onnx_save_path)
        print('success export yolov7.')
if __name__ == '__main__':
    parser = parse_arg()
    args = parser.parse_args()
    main(args=args)