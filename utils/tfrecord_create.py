import tensorflow as tf
import lxml.etree
import tqdm
import hashlib
import os
import cv2
import numpy as np
from .dataloader import get_classes

IMAGE_FEATURE_MAP = {
    # 'image/width': tf.io.FixedLenFeature([], tf.int64),
    # 'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    # 'image/source_id': tf.io.FixedLenFeature([], tf.string),
    # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    # 'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    # 'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    # 'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    # 'image/object/view': tf.io.VarLenFeature(tf.string),
}

# 解析xml
def parse_xml(xml):
    if not len(xml):
        return {xml.tag:xml.text}
    result = {}
    # 用递归方式去遍历
    for child in xml:
        child_result = parse_xml(child)
        if child.tag!='object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def create_tfrecord(annotation, class_map, data_dir, sub_folder='JPEGImages'):
    img_path = os.path.join(data_dir, sub_folder, annotation['filename'])
    img_raw = open(img_path,'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()
    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    views = []
    difficult_obj = []
    if 'object' in annotation:
        for obj in annotation['object']:
            difficult = bool(int(obj['difficult']))
            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_map[obj['name']])
            truncated.append(int(obj['truncated']))
            views.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example

# 解析tfrecord数据
def parse_tfrecord(tfrecord, class_table, size, max_boxes=100):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, size)
    # x_train = tf.image.resize_with_pad(x_train, size, size, 'nearest', antialias=True)

    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')
    labels = tf.cast(class_table.lookup(class_text), tf.float32)
    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)
    filenames = tf.stack([x['image/filename']])
    # correct box


    paddings = [[0, max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train

def load_tfrecord_dataset(file_pattern, class_file, size=(416, 416)):
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    '''
    tf.lookup.StaticHashTable：建立类别与数字的关联关系
    keys_tensor = tf.constant([1, 2])
    vals_tensor = tf.constant([3, 4])
    input_tensor = tf.constant([1, 5])
    table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
    print(table.lookup(input_tensor))
    output:tf.Tensor([ 3 -1], shape=(2,), dtype=int32)

    tf.lookup.TextFileInitializer：Table initializers from a text file.
    '''
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)

    # files = tf.data.Dataset.list_files(file_pattern)
    # dataset = files.flat_map(tf.data.TFRecordDataset)
    dataset = tf.data.TFRecordDataset(file_pattern)
    # debug function 
    # for ds in dataset:
    #     parse_tfrecord(ds, class_table, size)
    return dataset.map(lambda x: parse_tfrecord(x, class_table, size))


def visual_dataset(tfrecord_file,class_path, size, class_names):
    dataset = load_tfrecord_dataset(tfrecord_file, class_path, size)
    dataset = dataset.shuffle(512)
    for image, labels,filename in dataset:
        boxes = []
        scores = []
        classes = []
        for x1, y1, x2, y2, label in labels:
            if x1 == 0 and x2 == 0:
                continue

            boxes.append((x1, y1, x2, y2))
            scores.append(1)
            classes.append(label)
        nums = [len(boxes)]
        boxes = [boxes]
        scores = [scores]
        classes = [classes]
        img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite('./test.jpg', img)

def checkBox(box_set, max_x=416, max_y=416):
    new_box_x = 0
    new_box_y = 0
    if box_set[0]<0:
        new_box_x = 10
    elif box_set[0]>max_x:
        new_box_x = max_x-10
    else:
        new_box_x = box_set[0]
    
    if box_set[1]<0:
        new_box_y = 10
    elif box_set[1]>max_x:
        new_box_y = max_x-10
    else:
        new_box_y = box_set[1]
    
    return (new_box_x, new_box_y)

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))

        # 检查box的问题
        max_x = img.shape[0]
        max_y = img.shape[1]
        # x1y1 = checkBox(x1y1, max_x, max_y)
        # x2y2 = checkBox(x2y2, max_x, max_y)


        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img

def main_create_tfrecord(tfrecord_save_path, dataset_root, class_path, dataset_type):
    writer = tf.io.TFRecordWriter(tfrecord_save_path)
    classes_name = get_classes(classes_path=class_path)
    class_map = {name:id for id, name in enumerate(classes_name)}
    image_list = open(os.path.join(
        dataset_root, 'ImageSets', 'Main', f'{dataset_type}.txt')).read().splitlines()
    for name in tqdm.tqdm(image_list):
        xml_path = os.path.join(
            dataset_root, 'Annotations', name + '.xml')
        annotation_xml = lxml.etree.fromstring(open(xml_path).read().encode('utf-8'))
        annotation = parse_xml(annotation_xml)['annotation']
        # 创建example
        tf_example = create_tfrecord(annotation, class_map, data_dir='./villages')
        writer.write(tf_example.SerializeToString())
    writer.close()
    # visual_dataset(tfrecord_save_path, class_path, 416, classes_name)
    print('finish')


# 对图像数据进行处理
def transform_dataset(x_train, size):
    x_train = tf.image.resize(x_train, size)
    x_train = x_train/255
    return x_train

# 标签进行处理
# def transform_targets(true_boxes, input_shape, anchors, num_classes):

