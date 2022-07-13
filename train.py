from train import train_tiny, train_yolov4, train_yolox

type = 'yolox'

if type.upper() == 'YOLOX':
    train_yolox.main()
elif  type.upper() == 'YOLOV4':
    train_yolov4.main()
elif type.upper() == 'YOLOV4_TINY':
    train_tiny.main()
else:
    pass