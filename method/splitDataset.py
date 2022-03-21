from absl import app, flags, logging
from absl.flags import FLAGS
import os
from typing import List, Any
import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split

flags.DEFINE_string('jsonpath', '', 'path to label(*.json)')
flags.DEFINE_bool('isJSON', False, 'convert jsonFile or not')
flags.DEFINE_string('savepath','./village/', 'path to save directory')
flags.DEFINE_bool('Test', True, 'Create test set or not')

def main(_argv):
    # 1.标签路径
    labelme_path = FLAGS.jsonpath
    #原始labelme标注数据路径
    saved_path = FLAGS.savepath
    # 保存路径
    isUseTest=True#是否创建test集
    # 2.创建要求文件夹
    if not os.path.exists(saved_path + "Annotations"):
        os.makedirs(saved_path + "Annotations")
    if not os.path.exists(saved_path + "JPEGImages/"):
        os.makedirs(saved_path + "JPEGImages/")
    if not os.path.exists(saved_path + "ImageSets/Main/"):
        os.makedirs(saved_path + "ImageSets/Main/")
    # 3.获取待处理文件
    files = glob(labelme_path + "*.json")
    files = [i.replace("\\","/").split("/")[-1].split(".json")[0] for i in files]
    print(files)
    if FLAGS.isJSON:
        # 4.读取标注信息并写入 xml
        for json_file_ in files:
            json_filename = labelme_path + json_file_ + ".json"
            json_file = json.load(open(json_filename, "r", encoding="utf-8"))
            height, width, channels = cv2.imread('labelmedataset/images/' + json_file_ + ".jpg").shape
            with codecs.open(saved_path + "Annotations/" + json_file_ + ".xml", "w", "utf-8") as xml:
                xml.write('<annotation>\n')
                xml.write('\t<folder>' + 'AI' + '</folder>\n')
                xml.write('\t<filename>' + json_file_ + ".jpg" + '</filename>\n')
                xml.write('\t<source>\n')
                xml.write('\t\t<database>AGCIMAI</database>\n')
                xml.write('\t</source>\n')
                xml.write('\t<size>\n')
                xml.write('\t\t<width>' + str(width) + '</width>\n')
                xml.write('\t\t<height>' + str(height) + '</height>\n')
                xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
                xml.write('\t</size>\n')
                xml.write('\t\t<segmented>0</segmented>\n')
                for multi in json_file["shapes"]:
                    points = np.array(multi["points"])
                    labelName=multi["label"]
                    xmin = min(points[:, 0])
                    xmax = max(points[:, 0])
                    ymin = min(points[:, 1])
                    ymax = max(points[:, 1])
                    label = multi["label"]
                    if xmax <= xmin:
                        pass
                    elif ymax <= ymin:
                        pass
                    else:
                        xml.write('\t<object>\n')
                        xml.write('\t\t<name>' + labelName+ '</name>\n')
                        xml.write('\t\t<pose>Unspecified</pose>\n')
                        xml.write('\t\t<truncated>1</truncated>\n')
                        xml.write('\t\t<difficult>0</difficult>\n')
                        xml.write('\t\t<bndbox>\n')
                        xml.write('\t\t\t<xmin>' + str(int(xmin)) + '</xmin>\n')
                        xml.write('\t\t\t<ymin>' + str(int(ymin)) + '</ymin>\n')
                        xml.write('\t\t\t<xmax>' + str(int(xmax)) + '</xmax>\n')
                        xml.write('\t\t\t<ymax>' + str(int(ymax)) + '</ymax>\n')
                        xml.write('\t\t</bndbox>\n')
                        xml.write('\t</object>\n')
                        print(json_filename, xmin, ymin, xmax, ymax, label)
                xml.write('</annotation>')
        # 5.复制图片到 VOC2007/JPEGImages/下
        # image_files = glob("labelmedataset/images/" + "*.jpg")
        # print("copy image files to VOC007/JPEGImages/")
        # for image in image_files:
        #     shutil.copy(image, saved_path + "JPEGImages/")
    # 6.split files for txt
    txtsavepath = saved_path + "ImageSets/Main/"
    ftrainval = open(txtsavepath + '/trainval.txt', 'w')
    ftest = open(txtsavepath + '/test.txt', 'w')
    ftrain = open(txtsavepath + '/train.txt', 'w')
    fval = open(txtsavepath + '/val.txt', 'w')
    total_files = glob(saved_path+"/Annotations/*.xml")
    total_files = [i.replace("\\","/").split("/")[-1].split(".xml")[0] for i in total_files]
    trainval_files=[]
    test_files=[] 
    if isUseTest:
       trainval_files, test_files = train_test_split(total_files, test_size=0.2, random_state=55) 
    else: 
       trainval_files=total_files 
    for file in trainval_files: 
       ftrainval.write(file + "\n") 
    # split 
    train_files, val_files = train_test_split(trainval_files, test_size=0.2, random_state=55) 
    # train
    for file in train_files: 
       ftrain.write(file + "\n") 
    # val 
    for file in val_files: 
       fval.write(file + "\n")
    for file in test_files:
        print(file)
        ftest.write(file + "\n")
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:

        pass