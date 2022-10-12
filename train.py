import customerConf
import argparse

from train.train_tiny import yolov4tiny


# TODO: Add Command arg
def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    parser.add_argument(
        '--model', 
        help='YOLOV4, YOLOV4-TINY, YOLOV5 or YOLOX', 
        choices=['YOLOV4', 'YOLOV4-TINY', 'YOLOV5', 'YOLOX', 'YOLOV7'],
        default='YOLOV7', 
        type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.model.upper() == 'YOLOX':
        from train import yolox
        yolox(customerConf.YOLOXConfig)
    elif  args.model.upper() == 'YOLOV4':
        from train import yolov4
        yolov4(customerConf.YOLOV4Config)
    elif args.model.upper() == 'YOLOV4-TINY':
        from train import yolov4tiny
        yolov4tiny(customerConf.YOLOV4Config)
    elif args.model.upper() == 'YOLOV5':
        from train import yolov5
        yolov5(customerConf.YOLOV5Config)
    elif args.model.upper() == 'YOLOV7':
        from train import yolov7
        yolov7(customerConf.YOLOV7Config)
    else:
        pass