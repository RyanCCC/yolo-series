## Keras 实现Yolov4

#### 项目结构

``` python

+---logs：存放训练日志的文档
|---method：一些基础方法，如划分数据集训练集，生成训练集文档等
|---model_data：基础配置
|   +---simhei.ttf：字体
|   +---yolo_anchors.txt：预设置的anchorbox
|   +---yolo4_voc_weights.h5：VOC预训练权重
|   \---yolo4_weight.h5：COCO预训练权重
|---nets：Yolov4网络代码
|---result：推理结果保存的文件夹
|---utils：基础模块
\---datasets：数据集，以VOC数据集格式
    +---Annotations：数据集标注
    +---ImageSets
    |   \---Main：划分训练集、测试集、验证集的txt文档
    \---JPEGImages：图像
```

#### 执行步骤：

1. 生成训练集、测试集以及验证集：运行voc_annotation.py，注意路径设置

```python
base_path = './villages'
class_file = './villages/village.names'

数据集的文件结构：

|---villages
    +---Annotations
    +---ImageSets
        |---Main
           |---test.txt
           |---train.txt
           |---trainval.txt
           |---val.txt
    \---JPEGImages

```

2. 运行train.py文件，要注意一些路径的设置

```python
annotation_path = './villages/train.txt'
log_dir = 'logs/'
classes_path = 'villages/village.names'    
anchors_path = './model_data/yolo_anchors.txt'
weights_path = './model_data/yolo4_weight.h5'
save_model_name = 'village.h5'
input_shape = (416,416)
```