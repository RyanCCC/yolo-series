'''
Export model(YOLOV5、YOLOV5-V61、YOLOV7) to ONNX.

Usage:
    python ./export.py --weight ./voc_yolov5_l.pth --save_file ./yolov5_l_12.onnx --yolo yolov5
'''

import argparse
import os
import numpy as np

'''
Export pytorch to onnx

Usage:
     python .\export.py --yolo yolox --save_file ./test.onnx --opset 13 --weight .\yolox\checkpoints\yolox_s.pth
'''

def parse_arg():
    parser = argparse.ArgumentParser(description="Export YOLO model")
    parser.add_argument('--weight', type=str, help='model weight', default='./model/yolov5/village_detection_yolov5v61_s_2022_12_12.pth')
    parser.add_argument('--yolo', type=str, help='YOLO algorithm.', choices=['yolov5', 'yolov5-v61', 'yolov7', 'yolox'], default='yolov5-v61')
    parser.add_argument('--save_file', type=str, help='save onnx model name', default='./test.onnx')
    parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    return parser

def main(args):
    onnx_save_path = args.save_file
    opset = args.opset
    yolo_type = args.yolo
    weight = args.weight
    dynamic = False
    train = False
    simplify = False
    if args.dynamic:
        dynamic = True
    if args.train:
        train = False
    if args.simplify:
        simplify
    if yolo_type == 'yolov5':
        from yolov5 import export_model
        export_model(weights=weight, save_file=onnx_save_path, simplify=simplify, train=train, dynamic=dynamic, opset=opset)
    elif yolo_type == 'yolov5-v61':
        from yolov5v61 import export_model
        export_model(weights=weight, save_file=onnx_save_path, simplify=simplify, train=train, dynamic=dynamic, opset=opset)
    elif yolo_type == 'yolov7':
        from yolov7 import export_model
        export_model(weights=weight, save_file=onnx_save_path, simplify=simplify, train=train, dynamic=dynamic, opset=opset)
    elif yolo_type == 'yolox':
        from yolox import export_model
        export_model(weights=weight, save_file=onnx_save_path, simplify=simplify, train=train, dynamic=dynamic, opset=opset)
if __name__ == '__main__':
    parser = parse_arg()
    args = parser.parse_args()
    main(args=args)