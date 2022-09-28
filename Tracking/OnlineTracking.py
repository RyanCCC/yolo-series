'''
使用rstp协议读取海康摄像头
'''

import cv2
import queue
import threading
import os

'''
解决Python OpenCV 读取视频并抽帧出现error while decoding的问题:
https://stackoverflow.com/questions/49233433/opencv-read-errorh264-0x8f915e0-error-while-decoding-mb-53-20-bytestream
https://blog.stormbirds.cn/articles/2021/11/07/1636277190085.html
'''

q = queue.Queue()
src_path = "rtsp://admin:abcd1234@172.18.27.54:554/Streaming/Channels/101"

def Receive():
    print('start Reveive')
    cap = cv2.VideoCapture(src_path)
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)
    
def Display():
    print('Start Displaying')
    cv2.namedWindow("Output Video", 0)
    cv2.resizeWindow("Output Video", 1200, 1000)
    while True:
        if q.empty() is not True:
            frame = q.get()
            cv2.imshow("Output Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    p1=threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()
    