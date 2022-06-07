# YoloSeries

基于Tensorflow2实现YOLO系列的算法，当前实现了两个算法，一个是YOLOV4，另一个是YOLOX。后续将追踪关于YOLO的一些成果，并及时复现当中的算法。具体可以参考本人的博客：[目标检测2022最新进展](https://blog.csdn.net/u012655441/article/details/123552537)。

## YOLOV4 Implement by TF2

文件结构：

``` python

+---logs：存放训练日志的文档
|---method：一些基础方法，如划分数据集训练集，生成训练集文档等
|---data：基础配置
|   +---simhei.ttf：字体
|   +---yolo_anchors.txt：预设置的anchorbox
|   \---voc.names：存放类别名称，同理可换成customer类别名称
|---model：存放模型和权重
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

### 执行步骤：

1. 生成训练集、测试集以及验证集：运行voc_annotation.py，注意路径设置

```python
base_path = './villages'
class_file = './villages/village.names'

数据集的文件结构：

|---train_datasets
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
annotation_path = './train_datasets/train.txt'
log_dir = 'logs/'
classes_path = 'train_datasets/village.names'    
anchors_path = './data/yolo_anchors.txt'
weights_path = './data/yolo4_weight.h5'
save_model_name = 'village.h5'
input_shape = (416,416)
```

### MAP计算步骤

1. 统计测试集的groundtrue

```python

python get_gt_txt.py

```

2. 计算模型推理测试集的结果

```python

python get_dr_txt.py

```

3. 计算map的性能指标

```python

python get_map.py

```

## YOLOX

YOLOX的代码可以直接运行```train_yolox.py```即可。



## 参考

1. [YOLOV4代码实现](https://github.com/AlexeyAB/darknet)

2. [YOLOV4](./doc/yolov4.md)

3. [YOLOV4论文](https://arxiv.org/pdf/2004.10934.pdf)

4. [Object-Detection-Metrics](./doc/Object-Detection-Metrics.md)

5. [目标检测算法Yolov4详解](https://cloud.tencent.com/developer/article/1748630)

6. [mAP计算代码](https://github.com/Cartucho/mAP)

7. [YOLOV3-tf2](https://github.com/zzh8829/yolov3-tf2)

8. [YOLOX Code](https://github.com/Megvii-BaseDetection/YOLOX)

9. [YOLOX Paper](https://arxiv.org/abs/2107.08430)
