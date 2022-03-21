import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from nets.yolo import YOLO
import os
import config as sys_config

if __name__ == "__main__":
    yolo = YOLO(
        model_path=sys_config.model_path,
        anchors_path=sys_config.anchors_path,
        classes_path=sys_config.classes_path,
        score=sys_config.score,
        iou=sys_config.iou,
        max_boxes=sys_config.max_boxes,
        model_image_size=(sys_config.imagesize, sys_config.imagesize),
        letterbox_image=sys_config.letterbox_image
    )
    mode = sys_config.mode
    video_path      = sys_config.video_path
    video_save_path = sys_config.video_save_path
    video_fps       = sys_config.video_fps

    if mode == "predict":
        img = sys_config.image
        image = Image.open(img)
        r_image = yolo.detect_image(image)
        r_image.show()
    elif mode == 'testset':
        testfile = sys_config.test_txt_file
        with open(testfile, 'r') as f:
            testset = f.readlines()
        for img in testset:
            img = img.split()[0]
            image = Image.open(img)
            r_image = yolo.detect_image(image)
            resultname = os.path.basename(img)
            r_image.save(f'./result/{resultname}')
        print('开始跑')

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            ref,frame=capture.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(yolo.detect_image(frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        test_interval = 100
        img = Image.open('img/street.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video' or 'fps'.")
