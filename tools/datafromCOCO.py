from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
 
# 需要设置的路径
savepath="./todo/" 
img_dir=savepath+'images/'
anno_dir=savepath+'annotations/'
datasets_list=['val2017']
classes_names = ['dining table','tv'] 

my_annotation_dict = {
    'dining table':'diningtable',
    'tv':'tvmonitor'
}

'''
目录格式如下：
$COCO_PATH
----|annotations
----|train2017
----|val2017
----|test2017
'''
dataDir= r'D:/Datasets/COCO/'
 
headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""
 
tailstr = '''\
</annotation>
'''
 
# 检查目录是否存在，如果存在，先删除再创建，否则，直接创建
def mkr(path):
    if not os.path.exists(path):
        os.makedirs(path)  # 可以创建多级目录

def id2name(coco):
    classes=dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']]=cls['name']
    return classes
 
def write_xml(anno_path,head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        if obj[0] in my_annotation_dict.keys():
            f.write(objstr%(my_annotation_dict[obj[0]],obj[1],obj[2],obj[3],obj[4]))
        else:
            f.write(objstr%(obj[0],obj[1],obj[2],obj[3],obj[4]))
    f.write(tail)
 
 
def save_annotations_and_imgs(coco,dataset,filename,objs):
    #将图片转为xml，例:COCO_train2017_000000196610.jpg-->COCO_train2017_000000196610.xml
    dst_anno_dir = os.path.join(anno_dir, dataset)
    mkr(dst_anno_dir)
    anno_path=dst_anno_dir + '/' + filename[:-3]+'xml'
    img_path=dataDir+dataset+'/'+filename
    print("img_path: ", img_path)
    dst_img_dir = os.path.join(img_dir, dataset)
    mkr(dst_img_dir)
    dst_imgpath=dst_img_dir+ '/' + filename
    print("dst_imgpath: ", dst_imgpath)
    img=cv2.imread(img_path)
    #if (img.shape[2] == 1):
    #    print(filename + " not a RGB image")
     #   return
    shutil.copy(img_path, dst_imgpath)
 
    head=headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path,head, objs, tail)
 
 
def showimg(coco,dataset,img,classes,cls_id,show=True):
    global dataDir
    I=Image.open('%s/%s/%s'%(dataDir,dataset,img['file_name']))
    #通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        class_name=classes[ann['category_id']]
        if class_name in classes_names:
            print(class_name)
            if 'bbox' in ann:
                bbox=ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
                draw = ImageDraw.Draw(I)
                draw.rectangle([xmin, ymin, xmax, ymax])
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()
 
    return objs
 
for dataset in datasets_list:
    #./COCO/annotations/instances_train2017.json
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataset)
 
    #使用COCO API用来初始化注释数据
    coco = COCO(annFile)
 
    #获取COCO数据集中的所有类别
    classes = id2name(coco)
    print(classes)
    #[1, 2, 3, 4, 6, 8]
    classes_ids = coco.getCatIds(catNms=classes_names)
    print(classes_ids)
    for cls in classes_names:
        #获取该类的id
        cls_id=coco.getCatIds(catNms=[cls])
        img_ids=coco.getImgIds(catIds=cls_id)
        print(cls,len(img_ids))
        # imgIds=img_ids[0:10]
        for imgId in tqdm(img_ids[:500]):
            img = coco.loadImgs(imgId)[0]
            filename = img['file_name']
            # print(filename)
            objs=showimg(coco, dataset, img, classes,classes_ids,show=False)
            print(objs)
            save_annotations_and_imgs(coco, dataset, filename, objs)