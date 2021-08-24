import sys
import os
sys.path.append(os.getcwd())
import glob
import xml.etree.ElementTree as ET
import config as sys_config

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

image_ids = open(os.path.join(sys_config.test_txt, 'test.txt')).read().strip().split()

gt_folder = os.path.join(sys_config.result, sys_config.gt_folder_name)
if not os.path.exists(gt_folder):
    os.makedirs(gt_folder)

for image_id in image_ids:
    with open(os.path.join(gt_folder, image_id+".txt"), "w") as new_f:
        root = ET.parse( os.path.join(sys_config.dataset_base_path, "Annotations", image_id+".xml")).getroot()
        for obj in root.findall('object'):
            difficult_flag = False
            if obj.find('difficult')!=None:
                difficult = obj.find('difficult').text
                if int(difficult)==1:
                    difficult_flag = True
            obj_name = obj.find('name').text
            # classes_path = 'model_data/voc_classes.txt'
            # class_names = get_classes(classes_path)
            # if obj_name not in class_names:
            #     continue

            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text

            if difficult_flag:
                new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
            else:
                new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

print("Conversion completed!")
