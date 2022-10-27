import time
import cv2
import os
from PIL import Image
import numpy as np
from yolox import Inference_YOLOXModel
from cfg import *
# 使用yolov7
from yolov7 import Inference_YOLOV7Model

yolox = Inference_YOLOXModel(YOLOXConfig)
yolov7 = Inference_YOLOV7Model(YOLOV7Config)

'''
视频推理
'''

url = './video/Test3.mp4'

capture = cv2.VideoCapture(url)
fps = 0.0
# 定义编解码器并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('./output.mp4', fourcc, 20.0, size)

while True:
    try:
        t1 = time.time()
        if capture.isOpened():
            ref, frame = capture.read()
            # 获取视频的时间戳
            millseconds = capture.get(cv2.CAP_PROP_POS_MSEC)
            if frame is not None and ref:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(np.uint8(frame))
                frame = np.array(yolov7.detect(frame))
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

                fps  = ( fps + (1./(time.time()-t1)) ) / 2
                print("fps= %.2f"%(fps))
                frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(frame)
                cv2.imshow("video",frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        else:
            break
    except Exception as e:
        break

capture.release()
out.release()
cv2.destroyAllWindows()