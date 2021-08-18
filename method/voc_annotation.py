import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

base_path = './villages'
class_file = './villages/village.names'

sets=['train', 'val', 'test']
# classes = ["trashbin", "sign", "person", "car", "archway", "trashcan", "camera", "wc", "building", "pavilion"]

def convert_annotation(image_id, list_file):
    # huo
    with open(class_file) as f:
        classes = f.read().strip().split()

    annotation_file = f'Annotations/{image_id}.xml'
    annotation = os.path.join(base_path, annotation_file)
    in_file = open(annotation, encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = os.getcwd()

for image_set in sets:
    txt_path = f'ImageSets/Main/{image_set}.txt'
    txt_path = os.path.join(base_path, txt_path)
    image_ids = open(txt_path, encoding='utf-8').read().strip().split()
    save_path = os.path.join(base_path, f'{image_set}.txt')
    list_file = open(save_path, 'w', encoding='utf-8')
    for image_id in tqdm(image_ids):
        img_path = f'JPEGImages/{image_id}.jpg'
        list_file.write(os.path.join(base_path, img_path))
        convert_annotation(image_id, list_file)
        list_file.write('\n')
    list_file.close()
