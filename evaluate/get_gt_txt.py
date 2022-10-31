import sys
import os
sys.path.append(os.getcwd())
import xml.etree.ElementTree as ET
from cfg import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--testset',
        help='testset file',
        type=str)
    parser.add_argument(
        '--gt_folder',
        help='Ground True folder',
        default='./result/gt_folder',
        type=str
    )
    parser.add_argument(
        '--annotation',
        help='Annotation dataset base path',
        required=True,
        type=str
    )
    args = parser.parse_args()
    return args
    

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

if __name__=='__main__':
    args = parse_args()
    testset = args.testset
    image_ids = open(testset).read().strip().split()
    gt_folder = args.gt_folder
    if not os.path.exists(gt_folder):
        os.makedirs(gt_folder)
    for image_id in image_ids:
        with open(os.path.join(gt_folder, image_id+".txt"), "w") as new_f:
            root = ET.parse( os.path.join(args.annotation, image_id+".xml")).getroot()
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text
                difficult_flag = False
                if obj.find('difficult')!=None:
                    difficult = obj.find('difficult').text
                    if int(difficult)==1:
                        difficult_flag = True
                obj_name = obj.find('name').text

                area = (int(right) - int(left))*(int(bottom)-int(top))
                height = root.find('size').find('height').text
                width = root.find('size').find('width').text
                img_area = int(height)*int(width)*0.1

                if area<img_area:
                    filename = root.find('filename').text
                    difficult_flag = True

                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    print("Conversion completed!")
