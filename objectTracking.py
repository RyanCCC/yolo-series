import colorsys
import cv2
import numpy as np
from PIL import Image
from nets.yolo4 import YOLO
import os
import config as sys_config
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from utils import generate_detections as gdet
import config as sys_config

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path,encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


if __name__ == '__main__':
    video_path = sys_config.video_path
    
    # 加载yolo模型
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
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # init deep sort
    model_filename = './model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)
   

    cap = cv2.VideoCapture(video_path)
    n = 0
    classname = get_class(sys_config.classes_path)
    
    
    hsv_tuples = [(x / len(classname), 1., 1.)
                      for x in range(len(classname))]
    video_save_path = sys_config.video_save_path
    video_fps       = sys_config.video_fps
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
    while(True):
        ret, frame = cap.read()
        if frame is None:
            break
        image = Image.fromarray(frame ) 
        bboxes, scores, classes = yolo.deep_sort_track(image)
        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, len(bboxes)]

        # read in all classes in .names file
        class_names = get_class(sys_config.classes_path)
        names = []
        for i, c in list(enumerate(classes)):
            names.append(class_names[c])

        # 转换bbox的格式，从

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        result = np.asarray(frame)
        # result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Output Video", result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        out.write(result)
    
    cv2.destroyAllWindows()
