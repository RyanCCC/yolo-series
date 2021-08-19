# Yolov4

### Yolov4原理以及实现

#### YOLOV4介绍

Yolov4论文的abstract可以看出Yolov4结合了前人的好处，用了大量技巧提高目标检测的效率。其中包括：加权残差链接（WRC），跨阶段部分链接（CSP），跨小批量标准化（CmBN），自对抗训练（SAT），Mish激活，马赛克数据增强，DropBlock正则化，CIoU Loss等等。

![image](https://user-images.githubusercontent.com/27406337/130028721-43e82cf5-fff6-4830-b33a-33536d80afb6.png)

Yolov4达到的效果：

![image](https://user-images.githubusercontent.com/27406337/130029170-67be34d5-b9a9-4aac-ad21-0c988d60023f.png)

Yolov4的贡献如下：

1. Develope an efficient and powerful object detection model.
2. Verify the influence of state-of-the-art Bag-of-Freebies and Bag-of-Specials methods of object detection during the detector training
3. Modify state-of-the-art methods and make them more effecient and suitable for single GPU training, including CBN(Cross-iteration batch normalization), PAN(Path aggregation network for instance segmentation), Sam(CBAM: Convolutional block attention module).(即一些即插即用的模块，后面需要学习)

#### Yolov4框架原理

Yolov4框架主要从以下几个方面展开：目标检测`通用检测框架`，`CSPDarknet53`，`SPP结构`，`PAN结构`和`Yolov3`。

###### 目标检测通用框架

![image](https://user-images.githubusercontent.com/27406337/130031449-47c1282d-ace8-4971-93d0-7d60004dbb12.png)

To sum up, an ordinary object detector is composed ofseveral parts:
- Input: Image, Patches, Image Pyramid
- Backbones: VGG16 [68], ResNet-50 [26], SpineNet[12], EfficientNet-B0/B7 [75], CSPResNeXt50 [81], CSPDarknet53 [81]
- Neck:
  - Additional blocks: SPP [25], ASPP [5], RFB[47], SAM [85]
  - Path-aggregation blocks: FPN [44], PAN [49], NAS-FPN [17], Fully-connected FPN, BiFPN[77], ASFF [48], SFAM [98]
- Heads:
  - Dense Prediction (one-stage):
       - RPN [64], SSD [50], YOLO [61], RetinaNet[45] (anchor based)
       - CornerNet [37], CenterNet [13], MatrixNet[60], FCOS [78] (anchor free)
   - Sparse Prediction (two-stage):
       - Faster R-CNN [64], R-FCN [9], Mask RCNN [23] (anchor based)
       - RepPoints [87] (anchor free)
 作为one-stage的yolo网络主要由三个主要组件组成：
 1. Backbone：在不同图像细粒度上聚合并形成图像特征的卷积神经网络
 2. Neck：一系列混合和组合图像特征的网络层，并将图像特征传递到预测层
 3. Head：对图像特征进行预测，生成边界框并预测类别

Yolov4的网络结构图（来源：https://cloud.tencent.com/developer/article/1748630）：

![image](https://user-images.githubusercontent.com/27406337/130032435-26ae1571-dc14-4aac-9c04-6e366a4129bf.png)

Yolov4介绍两种训练推理的套路：

1.**Bag of freebies**：在训练上增加一些策略，达到更高的精度并且在测试的时候不会增加额外的时间策略，如图像增强，网络正则化，类别不平衡的处理方法。我的理解是提高检测速度。
2. **Bag og specials**：降低检测速度，提高精度。如增加模型感受野SPP，ASPP，RFB等，引入注意力机制Squeeze-and-Excitation(SE)、S怕条例SWISH等。

![image](https://user-images.githubusercontent.com/27406337/130035306-d5a3ffc1-b080-4de3-bfbc-589804f0a613.png)

###### CSPDarknet53

Yolov4对Darknet53进行改进，借鉴CSPNet(Cross Stage Partial Networks:跨阶段局部网络)。其解决了其他大型卷积神经网络框架Backbone中网络优化的梯度信息重复问题，将梯度的变化从头到尾地集成到特征图中，因此减少了模型的参数量和FLOPS(floating point operations per second)数值，既保证了推理速度和准确率，又减少了模型尺寸。

![image](https://user-images.githubusercontent.com/27406337/130034254-8d15727e-a23c-4c1c-8c19-974cd8ded3f7.png)



#### Backbone训练策略

#### Backbone推理策略

#### 检测头训练策略

#### 检测头推理策略

### 目标检测评价指标

### 仓库说明

仓库主要有以下三个分支：
1. Matser分支：介绍yolov4
3. Keras分支：使用keras+tensorflow1.0实现yolov4算法
4. tf2分支：使用tensorflow2 实现yolov4算法。tensorflow2的训练速度会快些

### 参考

[yolov4代码实现](https://github.com/AlexeyAB/darknet)

[yolov4论文](https://arxiv.org/pdf/2004.10934.pdf)

[Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)

[目标检测算法Yolov4详解](https://cloud.tencent.com/developer/article/1748630)



