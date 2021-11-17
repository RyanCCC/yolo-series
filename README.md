# Yolov4

### Yolov4原理以及实现

#### YOLOV4介绍

Yolov4论文的abstract可以看出Yolov4结合了前人的好处，用了大量技巧提高目标检测的效率。其中包括：加权残差链接（WRC），跨阶段部分链接（CSP），跨小批量标准化（CmBN），自对抗训练（SAT），Mish激活，马赛克数据增强，DropBlock正则化，CIoU Loss等等。可以看成一篇目标检测的综述，里面用到的Tricks需要查阅相关的论文才知道。技巧类的论文可以查看：[Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/abs/1902.04103)和[Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)。

![image](https://user-images.githubusercontent.com/27406337/130028721-43e82cf5-fff6-4830-b33a-33536d80afb6.png)

Yolov4达到的效果：

![image](https://user-images.githubusercontent.com/27406337/130029170-67be34d5-b9a9-4aac-ad21-0c988d60023f.png)

Yolov4的贡献如下：

1. Develope an efficient and powerful object detection model.
2. Verify the influence of state-of-the-art Bag-of-Freebies and Bag-of-Specials methods of object detection during the detector training
3. Modify state-of-the-art methods and make them more effecient and suitable for single GPU training, including CBN(Cross-iteration batch normalization), PAN(Path aggregation network for instance segmentation), Sam(CBAM: Convolutional block attention module).(即一些即插即用的模块，后面需要学习)

**YOLOv4模型 = CSPDarkNet53 + SPP + PANet(path-aggregation neck) + YOLOv3-head**

- SPP来源于Kaiming He的SPP Net，主要因为它显著增加了感受野，分离出最重要的上下文功能，并且几乎不降低网络操作速度。
- [PANet](https://arxiv.org/abs/1803.01534)主要是特征融合的改进。
- YOLOv3-head，因为是anchor-base方法，因此分类、回归分支没有改变。


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

考虑到几方面的平衡：输入网络分辨率/卷积层数量/参数数量/输出维度。一个模型的分类效果好不见得其检测效果就好，想要检测效果好需要以下几点：
- 更大的网络输入分辨率——用于检测小目标
- 更深的网络层——能够覆盖更大面积的感受野
- 更多的参数——更好的检测同一图像内不同size的目标

![image](https://user-images.githubusercontent.com/27406337/130037996-f045edf3-b5ea-41b5-8dcb-671a378735e8.png)


论文中给出CSPResNext50([CSPNET: A NEW BACKBONE THAT CAN ENHANCE LEARNING CAPABILITY OF CNN](https://arxiv.org/pdf/1911.11929v1.pdf))和CSPDarknet53对比：

![image](https://user-images.githubusercontent.com/27406337/130037751-cf9b3e42-cf0a-4ce5-b6f9-a939005eabc4.png)

- Up to the object size - allows viewing the entire object
- Up to network size - allows viewing the context around the object
- Exceeding the network size - increases the number of connections between the image point and the final activation

![image](https://user-images.githubusercontent.com/27406337/130041283-7a895d72-6d76-4d43-8244-89aaff4c577f.png)

###### SPP结构

![image](https://user-images.githubusercontent.com/27406337/130041428-1ecf089e-be02-494e-9963-57ed00a8ecc9.png)

###### PAN结构

![image](https://user-images.githubusercontent.com/27406337/130041568-f2167a7e-cc0f-4c93-8f38-8365ec490c69.png)


#### Backbone训练策略

1. 数据增强

  ![image](https://user-images.githubusercontent.com/27406337/130042785-76cd53c5-8d3e-4993-94bd-b2163dce2b82.png)

  - CutMix：[CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899v2.pdf)

  - Mosaic
    Mosaic是一种将4张训练图像合并成一张进行训练的数据增强方式。这增强了对正常背景之外的对象的检测，丰富检测物体的背景。此外，每个小批包含一个大的变化图像。
  
    ![image](https://user-images.githubusercontent.com/27406337/130042452-d2f40134-ba53-4761-9635-77df71aa9212.png)
    
2. [DropBlock正则化](https://arxiv.org/pdf/1810.12890.pdf)

   DropBlock方法的引入是为了克服Dropout随机丢弃特征的主要缺点，Dropout被证明是全连接网络的有效策略，但在特征空间相关的卷积层中效果不佳。DropBlock技术在称为块的相邻相关区域中丢弃特征。这样既可以实现生成更简单模型的目的，又可以在每次训练迭代中引入学习部分网络权值的概念，对权值矩阵进行补偿，从而减少过拟合。如下图：
   
   ![image](https://user-images.githubusercontent.com/27406337/130042912-57be2631-4e9f-40bc-9007-a7f765a25108.png)

3. 类标签平滑
  
    对于分类问题，特别是多分类问题，常常把向量转换成one-hot-vector，而one-hot带来的问题： 对于损失函数，我们需要用预测概率去拟合真实概率，而拟合one-hot的真实概率函数会带来两个问题：
    - 无法保证模型的泛化能力，容易造成过拟合；
    - 全概率和0概率鼓励所属类别和其他类别之间的差距尽可能加大，而由梯度有界可知，这种情况很难适应。会造成模型过于相信预测的类别。

    对预测有100%的信心可能表明模型是在记忆数据，而不是在学习。标签平滑调整预测的目标上限为一个较低的值，比如0.9。它将使用这个值而不是1.0来计算损失。这个概念缓解了过度拟合。说白了，这个平滑就是一定程度缩小label中min和max的差距，label平滑可以减小过拟合。所以，适当调整label，让两端的极值往中间凑凑，可以增加泛化性能。

#### Backbone推理策略

1. [Mish激活函数](https://arxiv.org/pdf/1908.08681.pdf)
  
    Mish激活函数的公式如下：
  
   ![image](https://user-images.githubusercontent.com/27406337/130162581-ca2599f4-47d6-4fad-a019-55a875739973.png)
  
   Mish是一个平滑的曲线，平滑的激活函数允许更好的信息深入神经网络，，从而得到更好的准确性和泛化；在负值的时候并不是完全截断，允许比较小的负梯度流入。
  
   ![image](https://user-images.githubusercontent.com/27406337/130162756-45fafde5-66a0-4366-afc9-cd636267a78f.png)

  
2. MiWRC策略（[BiFPN](https://arxiv.org/pdf/1911.09070.pdf)）

    ![image](https://user-images.githubusercontent.com/27406337/130162814-cd0e3977-220c-44e6-9ff5-cbecc67a5199.png)


#### 检测头训练策略

1. CIoU-loss
   
   - 经典IoU loss:
   
     ![image](https://user-images.githubusercontent.com/27406337/130163483-da5468a8-4d29-4593-a8e2-c845d0dde0b0.png)

   - GIoU：[Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630)

      经典IoU loss存在以下两个问题：
      
      - 预测框bbox和ground truth bbox如果没有重叠，IOU就始终为0并且无法优化。也就是说损失函数失去了可导的性质。

      - IOU无法分辨不同方式的对齐，例如方向不一致等，如下图所示，可以看到三种方式拥有相同的IOU值，但空间却完全不同。
      
      ![image](https://user-images.githubusercontent.com/27406337/130163539-1408c7f8-f095-45b6-afdb-c4dec07345ff.png)
      
      ![image](https://user-images.githubusercontent.com/27406337/130163934-bfbc8dbf-605e-4b83-8dd3-e4291619ec5c.png)
      
      ![image](https://user-images.githubusercontent.com/27406337/130164063-8ba4a846-507f-4813-8c53-b9a0e48a8e9d.png)
      
      算法流程：
      
      ![image](https://user-images.githubusercontent.com/27406337/130164340-9db81cac-0df0-427d-8bc0-885c3441a0f6.png)

      
   - DIoU：[dinstance IoU](https://arxiv.org/pdf/1911.08287.pdf)
      
      解决预测框与GT重叠时，GIoU退化成IoU，导致在预测框bbox和ground truth bbox包含的时候优化变得非常困难，特别是在水平和垂直方向收敛难。

      ![image](https://user-images.githubusercontent.com/27406337/130164616-9623c343-96e3-4e4b-bb4f-1819f4dd3f8d.png)

      
      ![image](https://user-images.githubusercontent.com/27406337/130163623-ecbd910c-824b-4307-9063-87891b758643.png)
      
      ![image](https://user-images.githubusercontent.com/27406337/130164637-e665e02f-23ae-4bba-b80b-86852013531f.png)
      
      ![image](https://user-images.githubusercontent.com/27406337/130164709-3c277681-d824-4fe8-8bf3-fbb91b43f081.png)

      
   - CIou：Complete IoU
      
      一个好的目标框回归损失应该考虑三个重要的几何因素：重叠面积，中心点距离，长宽比。GIoU为了归一化坐标尺度，利用IOU并初步解决了IoU为0无法优化的问题。然后DIoU损失在GIoU Loss的基础上考虑了边界框的重叠面积和中心点距离。所以还有最后一个点上面的Loss没有考虑到，即Anchor的长宽比和目标框之间的长宽比的一致性。
      
      ![image](https://user-images.githubusercontent.com/27406337/130163658-39269df9-12d9-4e96-9b34-dbbdb0c7142f.png)

2. CmBN策略
  
  CBN在计算当前时刻统计量时考虑前K个时刻统计量，以此扩大batch size操作。
  
  ![image](https://user-images.githubusercontent.com/27406337/130165308-7866b06d-700c-48c1-9352-dd0593572e67.png)

3. 自对抗训练(SAT)

  ![image](https://user-images.githubusercontent.com/27406337/130165514-1de0a146-bfc1-434d-9738-a88674eb13de.png)

4. [遗传算法优化超参](https://arxiv.org/pdf/2004.10934.pdf)

#### 检测头推理策略

1. SAM模块

  ![image](https://user-images.githubusercontent.com/27406337/130165696-052d75f7-9513-45fe-bd05-3b1698a13640.png)

2. DIoU-NMS

  ![image](https://user-images.githubusercontent.com/27406337/130165726-e09bc6a1-df33-4588-8a9e-895ec51b8667.png)


#### Yolov4实验

  ![image](https://user-images.githubusercontent.com/27406337/130165934-61594e35-7ba8-41c8-a367-e6f69d0541c3.png)

  ![image](https://user-images.githubusercontent.com/27406337/130165953-b55cea29-d9be-4ece-8c7b-3203b45b4b0f.png)

  ![image](https://user-images.githubusercontent.com/27406337/130165965-1cbb6b99-8ded-4361-8e6e-999f8058a2a9.png)
  
  ![image](https://user-images.githubusercontent.com/27406337/130165986-d733ff7f-243a-40c3-806f-dbb97ac52561.png)


### 目标检测评价指标(https://github.com/rafaelpadilla/Object-Detection-Metrics)

#### 准确率（Accuracy）

  正确分类的样本数除以样本总数，即 accuracy=正确预测的正反例数

#### 错误率（Error rate）

  错误率与正确率相反，描述被分类器错分的比例，即：
    
  错误率 = 1-准确率

#### 混淆矩阵（Confusion Matrix）

  混淆矩阵又被称为错误矩阵，在每个类别下，模型预测错误的结果数量以及错误预测类别和正确预测的数量都在这一矩阵下面显示出来。
    
  ![image](https://user-images.githubusercontent.com/27406337/130168328-e89ee068-6726-4626-a952-c07b051c374a.png)

#### 召回率（Recall）
    
    查全率，预测为正例的样本中正确的数量除以真正的Positive的数量，即Recall = TP/(TP+FN) = TP/P

#### 精确率（Precision）

  查准率，被分为正例的示例中实际为正例的比例，即：Precision = TP/(TP+FP)
    
  ![image](https://user-images.githubusercontent.com/27406337/130168701-b8da8275-5046-413e-b462-1fc63310be08.png)

#### PR曲线

  选取不同阈值时对应的精度和召回。总体趋势，精度越高，召回越低，当召回达到1时，对应概率分数最低的正样本。
    
  ![image](https://user-images.githubusercontent.com/27406337/130168796-682d5e7a-f525-4024-bdba-de80f7721f58.png)

#### 平均精度（Average-Precision AP）

  P-R曲线围起来的面积

#### F指标（F measure）

  ![image](https://user-images.githubusercontent.com/27406337/130169188-bbe0aa90-d67a-4fcf-ae41-e2aa45eb7955.png)

#### F指标

  ![image](https://user-images.githubusercontent.com/27406337/130169234-d5ddc7a5-3b30-4baa-85b4-2d6d90f1a384.png)

#### ROC曲线
    
  True Positive Rate ( TPR ) = TP / [ TP + FN] ，TPR代表能将正例分对的概率
    
  False Positive Rate( FPR ) = FP / [ FP + TN] ，FPR代表将负例错分为正例的概率
   
  ![image](https://user-images.githubusercontent.com/27406337/130169395-b392bdb6-9b46-4409-a266-f1c6a014acca.png)
    
  AUC则是ROC曲线围住的面积。

### 仓库说明

仓库主要有以下三个分支：

#### Matser分支

  介绍yolov4

#### Keras分支

  使用keras+tensorflow1.0实现yolov4算法
  
  conda环境：
  
  ```
  name: tf2
channels:
  - conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - defaults
dependencies:
  - backcall=0.2.0=pyh9f0ad1d_0
  - backports=1.0=py_2
  - backports.functools_lru_cache=1.6.4=pyhd8ed1ab_0
  - ca-certificates=2020.12.5=h5b45459_0
  - certifi=2020.12.5=py37h03978a9_1
  - colorama=0.4.4=pyh9f0ad1d_0
  - ipykernel=5.5.4=py37h7813e69_0
  - ipython=7.23.1=py37h7813e69_0
  - ipython_genutils=0.2.0=py_1
  - jedi=0.18.0=py37h03978a9_2
  - jupyter_client=6.1.12=pyhd8ed1ab_0
  - jupyter_core=4.7.1=py37h03978a9_0
  - libsodium=1.0.18=h8d14728_1
  - matplotlib-inline=0.1.2=pyhd8ed1ab_2
  - openssl=1.1.1k=h8ffe710_0
  - parso=0.8.2=pyhd8ed1ab_0
  - pickleshare=0.7.5=py_1003
  - pip=21.1.1=pyhd8ed1ab_0
  - prompt-toolkit=3.0.18=pyha770c72_0
  - pygments=2.9.0=pyhd8ed1ab_0
  - python=3.7.6=cpython_h60c2a47_6
  - python-dateutil=2.8.1=py_0
  - python_abi=3.7=1_cp37m
  - pywin32=300=py37hcc03f2d_0
  - pyzmq=22.0.3=py37hcce574b_1
  - setuptools=49.6.0=py37h03978a9_3
  - sqlite=3.35.5=h8ffe710_0
  - tornado=6.1=py37hcc03f2d_1
  - traitlets=5.0.5=py_0
  - vc=14.2=hb210afc_4
  - vs2015_runtime=14.28.29325=h5e1d092_4
  - wcwidth=0.2.5=pyh9f0ad1d_2
  - wheel=0.36.2=pyhd3deb0d_0
  - wincertstore=0.2=py37h03978a9_1006
  - zeromq=4.3.4=h0e60522_0
  - pip:
    - absl-py==0.12.0
    - aniso8601==9.0.1
    - astunparse==1.6.3
    - attrs==20.3.0
    - cachetools==4.2.2
    - cffi==1.14.5
    - chardet==4.0.0
    - click==7.1.2
    - cycler==0.10.0
    - decorator==4.4.2
    - dill==0.3.4
    - easydict==1.9
    - easyocr==1.4
    - flask==1.1.2
    - flask-docs==0.4.6
    - flask-login==0.5.0
    - flask-restful==0.3.9
    - flask-restplus==0.13.0
    - flatbuffers==1.12
    - future==0.18.2
    - gast==0.3.3
    - gevent==21.1.2
    - google-auth==1.30.0
    - google-auth-oauthlib==0.4.4
    - google-pasta==0.2.0
    - googleapis-common-protos==1.53.0
    - greenlet==1.1.0
    - grpcio==1.32.0
    - h5py==2.10.0
    - idna==2.10
    - imageio==2.9.0
    - imgaug==0.4.0
    - importlib-metadata==4.0.1
    - importlib-resources==5.2.0
    - itsdangerous==1.1.0
    - jinja2==2.11.3
    - joblib==1.0.1
    - jsonschema==3.2.0
    - keras==2.4.3
    - keras-preprocessing==1.1.2
    - kiwisolver==1.3.1
    - labelme2coco==0.1.2
    - lxml==4.6.3
    - markdown==3.3.4
    - markupsafe==1.1.1
    - matplotlib==3.4.1
    - networkx==2.5.1
    - numpy==1.19.5
    - oauthlib==3.1.0
    - onnxruntime==1.8.1
    - opencv-python==4.5.1.48
    - opt-einsum==3.3.0
    - pandas==1.2.4
    - pillow==8.2.0
    - pixellib==0.4.8
    - promise==2.3
    - protobuf==3.15.8
    - psycopg2==2.8.6
    - pyasn1==0.4.8
    - pyasn1-modules==0.2.8
    - pycparser==2.20
    - pyparsing==2.4.7
    - pyrsistent==0.17.3
    - python-bidi==0.4.2
    - pytz==2021.1
    - pywavelets==1.1.1
    - pyyaml==5.4.1
    - requests==2.25.1
    - requests-oauthlib==1.3.0
    - rsa==4.7.2
    - scikit-image==0.18.1
    - scikit-learn==0.24.2
    - scipy==1.6.3
    - seaborn==0.11.1
    - shapely==1.7.1
    - six==1.15.0
    - sklearn==0.0
    - tensorboard==2.5.0
    - tensorboard-data-server==0.6.1
    - tensorboard-plugin-wit==1.8.0
    - tensorflow==2.4.0
    - tensorflow-datasets==4.3.0
    - tensorflow-estimator==2.4.0
    - tensorflow-metadata==1.1.0
    - termcolor==1.1.0
    - theano==1.0.5
    - threadpoolctl==2.1.0
    - tifffile==2021.4.8
    - torch==1.9.0
    - torchvision==0.10.0
    - tqdm==4.61.0
    - typing-extensions==3.7.4.3
    - urllib3==1.25.8
    - werkzeug==0.16.1
    - wrapt==1.12.1
    - zipp==3.4.1
    - zope-event==4.5.0
    - zope-interface==5.4.0
prefix: D:\soft\Anaconda\envs\tf2

  ```
  
#### tf2分支

  使用tensorflow2 实现yolov4算法。tensorflow2的训练速度会快些
  
  Conda环境
  
  ```
  name: tf1
channels:
  - conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - defaults
dependencies:
  - backports=1.0=py_2
  - backports.functools_lru_cache=1.6.4=pyhd8ed1ab_0
  - certifi=2020.12.5=py36ha15d459_1
  - colorama=0.4.4=pyh9f0ad1d_0
  - ipykernel=5.5.4=py36hfacbf0b_0
  - ipython=5.8.0=py36_1
  - ipython_genutils=0.2.0=py_1
  - jupyter_client=6.1.12=pyhd8ed1ab_0
  - jupyter_core=4.7.1=py36ha15d459_0
  - libsodium=1.0.18=h8d14728_1
  - pickleshare=0.7.5=py_1003
  - prompt_toolkit=1.0.15=py_1
  - pygments=2.9.0=pyhd8ed1ab_0
  - python=3.6.13=h39d44d4_0_cpython
  - python-dateutil=2.8.1=py_0
  - python_abi=3.6=1_cp36m
  - pywin32=300=py36h68aa20f_0
  - pyzmq=22.0.3=py36h1d5d788_1
  - setuptools=49.6.0=py36ha15d459_3
  - simplegeneric=0.8.1=py_1
  - tornado=6.1=py36h68aa20f_1
  - traitlets=4.3.3=py36h9f0ad1d_1
  - vc=14.2=hb210afc_4
  - vs2015_runtime=14.28.29325=h5e1d092_4
  - wcwidth=0.2.5=pyh9f0ad1d_2
  - wheel=0.36.2=pyhd3deb0d_0
  - wincertstore=0.2=py36ha15d459_1006
  - zeromq=4.3.4=h0e60522_0
  - pip:
    - absl-py==0.12.0
    - alembic==1.5.8
    - aniso8601==9.0.1
    - astor==0.8.1
    - attrs==20.3.0
    - augmentor==0.2.8
    - bleach==1.5.0
    - cached-property==1.5.2
    - cachetools==4.2.2
    - chardet==4.0.0
    - click==7.1.2
    - cycler==0.10.0
    - cython==0.29.24
    - dataclasses==0.8
    - decorator==4.4.2
    - docopt==0.6.2
    - easydict==1.9
    - easyocr==1.4
    - faker==8.1.0
    - fire==0.4.0
    - flasgger==0.9.5
    - flask==1.1.2
    - flask-ckeditor==0.4.4.1
    - flask-login==0.5.0
    - flask-migrate==2.7.0
    - flask-moment==0.11.0
    - flask-restplus==0.13.0
    - flask-share==0.1.1
    - flask-sqlalchemy==2.5.1
    - flask-uploads==0.2.1
    - flask-wtf==0.14.3
    - flatbuffers==1.12
    - future==0.18.2
    - gast==0.4.0
    - google-api-core==1.31.0
    - google-api-python-client==2.14.0
    - google-auth==1.33.1
    - google-auth-httplib2==0.1.0
    - google-pasta==0.2.0
    - googleapis-common-protos==1.53.0
    - greenlet==1.0.0
    - grpcio==1.37.0
    - h5py==2.10.0
    - html5lib==0.9999999
    - httplib2==0.19.1
    - idna==2.10
    - imageio==2.9.0
    - imantics==0.1.12
    - imgaug==0.4.0
    - importlib-metadata==3.10.0
    - itsdangerous==1.1.0
    - jinja2==2.11.3
    - joblib==1.0.1
    - jsonschema==3.2.0
    - keras==2.1.5
    - keras-applications==1.0.8
    - keras-preprocessing==1.1.2
    - keras2onnx==1.7.0
    - kiwisolver==1.3.1
    - labelme2coco==0.1.2
    - lxml==4.6.3
    - mako==1.1.4
    - markdown==3.3.4
    - markupsafe==1.1.1
    - matplotlib==3.3.4
    - mistune==0.8.4
    - mysqlclient==2.0.3
    - networkx==2.5.1
    - numpy==1.19.5
    - oauth2client==4.1.3
    - onnx==1.9.0
    - onnxconverter-common==1.8.1
    - onnxruntime==1.8.1
    - opencv-python==4.5.1.48
    - packaging==21.0
    - pandas==1.1.5
    - pillow==8.2.0
    - pip==21.0.1
    - pipreqs==0.4.10
    - pixellib==0.6.1
    - protobuf==3.15.7
    - psycopg2==2.8.6
    - pyasn1==0.4.8
    - pyasn1-modules==0.2.8
    - pycocotools==2.0.2
    - pydrive==1.3.1
    - pymysql==1.0.2
    - pyparsing==2.4.7
    - pyrsistent==0.17.3
    - pytesseract==0.3.8
    - python-bidi==0.4.2
    - python-editor==1.0.4
    - pytils==0.3
    - pytz==2021.1
    - pywavelets==1.1.1
    - pyyaml==5.4.1
    - redis==3.5.3
    - requests==2.25.1
    - rsa==4.7.2
    - scikit-image==0.17.2
    - scikit-learn==0.24.2
    - scipy==1.5.4
    - shapely==1.7.1
    - six==1.15.0
    - sklearn==0.0
    - sqlalchemy==1.4.11
    - tensorboard==1.14.0
    - tensorflow==1.14.0
    - tensorflow-estimator==1.14.0
    - termcolor==1.1.0
    - text-unidecode==1.3
    - textdistance==4.2.1
    - tf2onnx==1.9.1
    - threadpoolctl==2.1.0
    - tifffile==2020.9.3
    - torch==1.9.0
    - torchaudio==0.9.0
    - torchsummary==1.5.1
    - torchvision==0.10.0
    - tqdm==4.60.0
    - typing-extensions==3.7.4.3
    - uritemplate==3.0.1
    - urllib3==1.26.4
    - werkzeug==0.16.1
    - wrapt==1.12.1
    - wtforms==2.3.3
    - xmljson==0.2.1
    - yarg==0.1.9
    - zipp==3.4.1
prefix: D:\soft\Anaconda\envs\tf1

  ```

### 参考

[yolov4代码实现](https://github.com/AlexeyAB/darknet)

[yolov4论文](https://arxiv.org/pdf/2004.10934.pdf)

[Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)

[目标检测算法Yolov4详解](https://cloud.tencent.com/developer/article/1748630)

[mAP计算代码](https://github.com/Cartucho/mAP)

[Yolov3-tf2](https://github.com/zzh8829/yolov3-tf2)



