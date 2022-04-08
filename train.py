from train import train, train_tiny, train_yolox

type = 'yolox'

if type.upper() == 'YOLOX':
    train_yolox.main()
elif  type.upper() == 'YOLOV4':
    train.main()
elif type.upper() == 'YOLOV4_TINY':
    train_tiny.main()
else:
    pass