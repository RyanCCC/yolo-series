import customerConf
import argparse

from yolov7.nets import yolov7


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
        from train import train_yolox
        train_yolox.main(customerConf.YOLOXConfig)
    elif  args.model.upper() == 'YOLOV4':
        from train import train_yolov4
        train_yolov4.main(customerConf.YOLOV4Config)
    elif args.model.upper() == 'YOLOV4-TINY':
        from train import train_tiny
        train_tiny.main(customerConf.YOLOV4Config)
    elif args.model.upper() == 'YOLOV5':
        from train import train_yolov5
        train_yolov5.train(customerConf.YOLOV5Config)
    elif args.model.upper() == 'YOLOV7':
        from train import train_yolov7
        train_yolov7.train(customerConf.YOLOV7Config)
    else:
        pass