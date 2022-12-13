"""
Export a YOLOV5 Pytorch model to ONNX
"""
import os
import subprocess
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from .predict_yolov5 import Inference_YOLOV5Model
from cfg import YOLOV5Config

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def export_onnx(model, file, opset, train, dynamic, simplify, prefix='\033[91m'):
    # YOLOv5 ONNX export
    try:
        import onnx
        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        im = torch.zeros(1, 3, *YOLOV5Config.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        torch.onnx.export(
            model, 
            im, 
            file, 
            verbose=False, 
            opset_version=opset,
            training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                                        } if dynamic else None)

        # Checks
        model_onnx = onnx.load(file)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        # Simplify
        if simplify:
            try:
                import onnxsim

                print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, file)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
        print(f'{prefix} export success, saved as {file}')
        print(f"{prefix} run --dynamic ONNX model inference with: 'python detect.py --weights {file}'")
    except Exception as e:
        print(f'{prefix} export failure: {e}')


def export_model(
    weights,
    save_file,
    simplify=False,
    include=('onnx',),
    train=False, # model.train() mode
    dynamic = False, # onnx:dynamic axes
    opset = 12, #ONNX: opset version
    ):
    yolov5 = Inference_YOLOV5Model(YOLOV5Config, weights, export=True).net
    include = [x.lower() for x in include]
    if 'onnx' in include:
        print('Exporting onnx model....')
        export_onnx(yolov5, save_file, opset, train, dynamic, simplify)
    