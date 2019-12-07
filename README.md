<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [**InsightFace算法学习**](#insightface算法学习)
	- [开源仓库](#开源仓库)
		- [识别算法配置](#识别算法配置)
	- [算法测试](#算法测试)
		- [训练数据](#训练数据)
		- [测试模型](#测试模型)
	- [相关算法](#相关算法)
	- [数据集](#数据集)
	- [参考文献](#参考文献)

<!-- /TOC -->

# **InsightFace算法学习**

## 开源仓库
- **InsightFace:** [https://github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)<br>
- **作者演讲：**[https://www.bilibili.com/video/av54356295?t=785](https://www.bilibili.com/video/av54356295?t=785)

### 识别算法配置
1. 安装mxnet
````
pip install mxnet-cu80 #or mxnet-cu90 or mxnet-cu100
````
2. 下载insightface算法仓库
````
git clone --recursive https://github.com/deepinsight/insightface.git
````
3. 下载(MS1MV2-Arcface)数据集，数据集里包含以下内容
````
faces_emore/
       train.idx
       train.rec
       property
       lfw.bin
       cfp_fp.bin
       agedb_30.bin
````
4. 复制配置文件
````
cp sample_config.py config.py
vim config.py # edit dataset path etc..
````
5. 配置环境变量
````
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
````
6. 模型训练
- 训练LResNet100E-IR网络，损失函数为ArcFace。
````
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r100 --loss arcface --dataset emore
````
- 训练LResNet50E-IR网络，损失函数为CosineFace。
````
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r50 --loss cosface --dataset emore
````
- 训练MobileFaceNet网络，损失函数为Softmax。
````
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network y1 --loss softmax --dataset emore
````
- Fine tune MobileFaceNet网络, 损失函数改为Triplet loss。
````
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network mnas05 --loss triplet --lr 0.005 --pretrained ./models/y1-softmax-emore,1
````
多GPU训练可以使用``train_parall.py``文件进行多GPU加速。

- **<font color ="red" size=4 face="TimesNewRoman">*人脸识别为一分类网络，insight face训练先在大数据集上使用ArcFace损失函数做分类训练，然后再使用Triplet损失函数进行微调提高识别精度。*</font>**

## 算法测试

### 训练数据

1. 直接训练(MS1MV2-Arcface)数据集，基于MobileFaceNet网络和ArcFace损失函数进行训练。
````
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network y1 --loss arcface --dataset emore
````
- 训练配置<br>
作者的训练配置如下所示：
````
Namespace(batch_size=512, beta=1000.0, beta_freeze=0, beta_min=5.0, bn_mom=0.9, ckpt=1, ctx_num=4, cutoff=0, data_dir='/cache/jiaguo/faces_ms1mi_112x112', easy_margin=0, emb_size=512, end_epoch=100000, fc7_wd_mult=1.0, gamma=0.12, image_channel=3, image_h=112, image_w=112, loss_type=5, lr=0.1, lr_steps='100000,140000,160000', margin=4, margin_a=1.0, margin_b=0.2, margin_m=0.3, margin_s=64.0, max_steps=0, mom=0.9, network='r100', num_classes=85742, num_layers=100, per_batch_size=128, power=1.0, prefix='../models2/model-r100-ii/model', pretrained='', rand_mirror=1, rescale_threshold=0, scale=0.9993, target='lfw,cfp_fp,agedb_30', use_deformable=0, verbose=2000, version_act='prelu', version_input=1, version_output='E', version_se=0, version_unit=3, wd=0.0005)
````
- 训练结果<br>
作者得到的训练结果如下所示：
````
testing verification..
(12000, 512)
infer time 21.69233
[lfw][168000]XNorm: 22.172497
[lfw][168000]Accuracy-Flip: 0.99783+-0.00269
testing verification..
(14000, 512)
infer time 24.988244
[cfp_fp][168000]XNorm: 21.383092
[cfp_fp][168000]Accuracy-Flip: 0.98271+-0.00569
testing verification..
(12000, 512)
infer time 21.44195
[agedb_30][168000]XNorm: 22.695239
[agedb_30][168000]Accuracy-Flip: 0.98233+-0.00716
[168000]Accuracy-Highest: 0.98283
````

### 测试模型

- **MegaFace测试**<br>
需要安装的依赖项：
```
tbb2 opencv2.4
```
如果高版本cuda不支持`opencv2.4`，将`FindCUDA.cmake`替换为最新版的[FindCUDA.cmake](https://github.com/opencv/opencv/blob/master/cmake/FindCUDA.cmake)，如果不支持`compute_20`，将`OpenCVDetectCUDA.cmake`替换为最新版的[OpenCVDetectCUDA.cmake](https://github.com/opencv/opencv/blob/master/cmake/OpenCVDetectCUDA.cmake)，
下载MegaFace的评估工具[devkit.tar.gz](http://megaface.cs.washington.edu/dataset/download/content/devkit.tar.gz),
从[网盘](https://pan.baidu.com/s/1h4ezfwJiXClbZDdg1RX0MQ)中下载MegaFace测试数据*megaface_testpack_v1.0.zip*，解压后文件夹中包含的数据有
````
facescrub_images/
megaface_images/
facescrub_lst
facescrub_lst_all
facescrub_noises.txt
facescrub_noises_empty.txt
megaface_lst
megaface_noises.txt
megaface_noises_empty.txt
````
在工程的`./Evaluation/Megaface/`文件夹中，运行`./run.sh`文件，测试模型在MegaFace数据集上的识别精度。运行`./run.sh`前，先修改devkit的路径`DEVKIT`,将`DEVKIT="/raid5data/dplearn/megaface/devkit/experiments"`改为`devkit/experiments`实际所在的位置，修改后，可以得到模型在MegaFace数据集上的识别精度，测试模型需要花费较长时间。
````
Done matching! Score matrix size: 3530 1000000
Saving to ../../mx_results/otherFiles/facescrub_megaface_r100ii_1000000_1.bin
Computing test results with 1000000 images for set 1
Loaded 3530 probes spanning 80 classes
Loading from ../../mx_results/otherFiles/facescrub_facescrub_r100ii.bin
Probe score matrix size: 3530 3530
distractor score matrix size: 3530 1000000
Done loading. Time to compute some stats!
Finding top distractors!
Done sorting distractor scores
Making gallery!
Done Making Gallery!
Allocating ranks (1000080)
Rank 1: 0.983584
````



## 相关算法
- **人脸检测：** RetinaFace<br>
- **人脸对齐：** Dense U-Net<br>
- **人脸识别：** ArcFace<br>

## 数据集
- **LFW:** [http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)
- **CFP:** [http://www.cfpw.io/index.html](http://www.cfpw.io/index.html)
- **AgeDB** [https://ibug.doc.ic.ac.uk/resources/agedb/](https://ibug.doc.ic.ac.uk/resources/agedb/)
- **MegaFace:** [http://megaface.cs.washington.edu/](http://megaface.cs.washington.edu/)
- **MS-Celeb-1M:** [https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)


## 参考文献
[1] **ArcFace:** Additive Angular Margin Loss for Deep Face Recognition, Jiankang Deng, Jia Guo, Niannan Xue,
Stefanos Zafeiriou, [https://arxiv.org/abs/1801.07698](https://arxiv.org/abs/1801.07698)<br>
[2] **RetinaFace:** Single-stage Dense Face Localisation in the Wild, Jiankang Deng, Jia Guo, Yuxiang Zhou,
Jinke Yu, Irene Kotsia, Stefanos Zafeiriou, [https://arxiv.org/abs/1905.00641](https://arxiv.org/abs/1905.00641)<br>
[3] Stacked **Dense U-Nets** with Dual Transformers for Robust Face Alignment, Jia Guo, Jiankang Deng,
Niannan Xue, Stefanos Zafeiriou, [https://arxiv.org/abs/1812.01936](https://arxiv.org/abs/1812.01936)<br>
