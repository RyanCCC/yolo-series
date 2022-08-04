from train import train_tiny, train_yolov4, train_yolov5, train_yolox
import config
import argparse


# TODO: Add Command arg
def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    parser.add_argument('--model', help='YOLOV4, YOLOV4-TINY, YOLOV5 or YOLOX', default='yolov4', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.model.upper() == 'YOLOX':
        train_yolox.main(config.YOLOXConfig)
    elif  args.model.upper() == 'YOLOV4':
        train_yolov4.main(config.YOLOV4Config)
    elif args.model.upper() == 'YOLOV4-TINY':
        train_tiny.main(config.YOLOV4Config)
    elif args.model.upper() == 'YOLOV5':
        train_yolov5.train(config.YOLOV5Config)
    else:
        pass