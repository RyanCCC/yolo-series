import cfg
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    parser.add_argument(
        '--model', 
        help='YOLOV4, YOLOV4-TINY, YOLOV5, YOLOX, YOLOV7...', 
        choices=['YOLOV4', 'YOLOV4-TINY', 'YOLOV5','YOLOV5-V61', 'YOLOX', 'YOLOV7'],
        default='YOLOV5-V61', 
        type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.model.upper() == 'YOLOX':
        from yolox import yolox
        yolox(cfg.YOLOXConfig)
    elif  args.model.upper() == 'YOLOV4':
        from yolov4 import yolov4
        yolov4(cfg.YOLOV4Config)
    elif args.model.upper() == 'YOLOV4-TINY':
        from yolov4 import yolov4_tiny
        yolov4_tiny(cfg.YOLOV4TinyConfig)
    elif args.model.upper() == 'YOLOV5':
        from yolov5 import yolov5
        yolov5(cfg.YOLOV5Config)
    elif args.model.upper() == 'YOLOV5-V61':
        from yolov5v61 import yolov5
        yolov5(cfg.YOLOV5Config)
    elif args.model.upper() == 'YOLOV7':
        from yolov7 import yolov7
        yolov7(cfg.YOLOV7Config)
    else:
        pass