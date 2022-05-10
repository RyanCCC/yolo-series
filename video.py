import time
import cv2
import os
from PIL import Image
import numpy as np
from predict_yolox import detect, build_model,get_classes
import config



capture = cv2.VideoCapture(0)
fps = 0.0
input_shape = [640,640]
model_path = './model/village_yolox_0510.h5'
classes_path = config.classes_path
class_names = get_classes(classes_path)
num_classes = len(class_names)
model = build_model(model_path,input_shape, class_names, letterbox_image=True)
# 定义编解码器并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('./output.mp4', fourcc, 20.0, size)

while True:
    t1 = time.time()
    ref, frame = capture.read()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)q
        frame = Image.fromarray(np.uint8(frame))
        frame = np.array(detect(frame, input_shape, model, classes_path))
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %.2f"%(fps))
        frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

        cv2.imshow("video",frame)
        if cv2.waitKey(1) == ord('q'):
            break
out.release()
capture.release()
out.release()
cv2.destroyAllWindows()