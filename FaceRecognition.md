<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [**InsightFace算法配置**](#insightface算法配置)
	- [开源仓库](#开源仓库)
		- [识别算法配置](#识别算法配置)
		- [模型训练](#模型训练)
		- [模型评估](#模型评估)
	- [评估结果](#评估结果)
	- [数据集](#数据集)

<!-- /TOC -->

# **InsightFace算法配置**

## 开源仓库
- **InsightFace:** [https://github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)<br>
- **作者演讲：**[https://www.bilibili.com/video/av54356295?t=785](https://www.bilibili.com/video/av54356295?t=785)

### 识别算法配置
**1. 安装mxnet**
````
pip install mxnet-cu80 #or mxnet-cu90 or mxnet-cu100
````
**2. 下载insightface算法仓库**
````
git clone --recursive https://github.com/deepinsight/insightface.git
````
**3. 下载(MS1MV2-Arcface)数据集<br>**
数据集下载地址在：[https://pan.baidu.com/s/1S6LJZGdqcZRle1vlcMzHOQ](https://pan.baidu.com/s/1S6LJZGdqcZRle1vlcMzHOQ)，数据集里包含以下内容
````
faces_emore/
       train.idx
       train.rec
       property
       lfw.bin
       cfp_ff.bin
       cfp_fp.bin
       agedb_30.bin
       calfw.bin
       cplfw.bin
       vgg2_fp.bin
````
**4. 复制配置文件**
````
cp sample_config.py config.py
vim config.py # edit dataset path etc..
````
- **<font color ="red" size=4 face="TimesNewRoman">*如果后面需要Fine Tune模型，需要将`config.py`的`config.ckpt_embedding`这一行的值改为`False`，这样可以保存网络的fc7层的权重，否则不会保存，不保存就会从头开始重新训练。*</font>** <br>
- *可选：将`config.py`文件中的这一行`dataset.emore.val_targets = ['lfw', 'cfp_fp', 'agedb_30']`修改为：`dataset.emore.val_targets = ['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30']`，在训练评估时可以同时评估cpf_ff数据集，想看模型在该数据集上的准确率可以加上，该数据集上的准确率与LFW的准确率接近*。

**5. 配置环境变量**
````
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
````
### 模型训练

- **<font color ="red" size=4 face="TimesNewRoman">*在训练前先确定模型训练的学习率，测试一下学习率为0.1，0.01和0.005等条件下的训练速度，选择准确率增长较快的学习率进行训练！！！*</font>**<br>

**1. 训练命令**<br>

- 训练LResNet100E-IR网络，损失函数为ArcFace。
````
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r100 --loss arcface --dataset emore  2>&1 | tee log.txt
````
- 训练LResNet50E-IR网络，损失函数为CosineFace。
````
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r50 --loss cosface --dataset emore  2>&1 | tee log.txt
````
- 训练MobileFaceNet网络，损失函数为Softmax。
````
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network y1 --loss softmax --dataset emore  2>&1 | tee log.txt
````
- Fine tune MobileFaceNet网络, 损失函数改为Triplet loss。
````
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network mnas05 --loss triplet --lr 0.005 --pretrained ./models/y1-softmax-emore,1  2>&1 | tee log.txt
````
- **<font color ="red" size=4 face="TimesNewRoman">*训练Triplet Loss时，需要在`train.py`文件中的这一行*</font>**<br>
```
model = mx.mod.Module(
		context=ctx,
		symbol=sym,
)
```
- **<font color ="red" size=4 face="TimesNewRoman">*后面增加*</font>**<br>
```
model.bind([("data", (args.batch_size, args.image_channel, image_size[0], image_size[1]))], [("softmax_label", (args.batch_size,))])
```
- **<font color ="red" size=4 face="TimesNewRoman">*这一句话，否则训练会报错！！！*</font>**<br>

多GPU训练可以使用``train_parall.py``文件进行多GPU加速。
作者的训练配置如下所示：每张卡上的batch size为128，共使用4张卡进行训练，故batch size为512。
````
Namespace(batch_size=512, beta=1000.0, beta_freeze=0, beta_min=5.0, bn_mom=0.9, ckpt=1, ctx_num=4, cutoff=0, data_dir='/cache/jiaguo/faces_ms1mi_112x112', easy_margin=0, emb_size=512, end_epoch=100000, fc7_wd_mult=1.0, gamma=0.12, image_channel=3, image_h=112, image_w=112, loss_type=5, lr=0.1, lr_steps='100000,140000,160000', margin=4, margin_a=1.0, margin_b=0.2, margin_m=0.3, margin_s=64.0, max_steps=0, mom=0.9, network='r100', num_classes=85742, num_layers=100, per_batch_size=128, power=1.0, prefix='../models2/model-r100-ii/model', pretrained='', rand_mirror=1, rescale_threshold=0, scale=0.9993, target='lfw,cfp_fp,agedb_30', use_deformable=0, verbose=2000, version_act='prelu', version_input=1, version_output='E', version_se=0, version_unit=3, wd=0.0005)
````

**2. 训练结果**<br>
作者在LFW、CFP和AgeDB30数据集上得到的训练结果如下所示：
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

### 模型评估
**1. MegaFace数据集评估**<br>
需要安装的依赖项：
```
opencv2.4
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

**2. LFW, CFP, AgeDB数据集评估**<br>
进入`./recognition/eval/`文件夹，输入以下命令，使用`verification.py`文件进行评估。需要指定模型所在的文件夹和评估数据所在的文件夹。
````
python verification.py --model ../../models/model-r100-ii/model --data-dir ../../datasets/faces_emore/
````
得到的结果如下所示：
````
(12000, 512)
infer time 108.986159
[lfw]XNorm: 22.132480
[lfw]Accuracy: 0.00000+-0.00000
[lfw]Accuracy-Flip: 0.99767+-0.00281
Max of [lfw] is 0.99767
testing verification..
(14000, 512)
infer time 121.617964
[cfp_ff]XNorm: 21.077436
[cfp_ff]Accuracy: 0.00000+-0.00000
[cfp_ff]Accuracy-Flip: 0.99843+-0.00162
Max of [cfp_ff] is 0.99843
testing verification..
(14000, 512)
infer time 122.128096
[cfp_fp]XNorm: 21.340035
[cfp_fp]Accuracy: 0.00000+-0.00000
[cfp_fp]Accuracy-Flip: 0.98271+-0.00559
Max of [cfp_fp] is 0.98271
testing verification..
(12000, 512)
infer time 104.282227
[agedb_30]XNorm: 22.654594
[agedb_30]Accuracy: 0.00000+-0.00000
[agedb_30]Accuracy-Flip: 0.98250+-0.00712
Max of [agedb_30] is 0.98250
````

- **<font color ="red" size=4 face="TimesNewRoman">*人脸识别为一分类网络，insight face训练先在大数据集上使用ArcFace损失函数做分类训练，然后再使用Triplet损失函数进行微调提高识别精度。*</font>**

## 评估结果
**1. 作者开源模型**
<center>模型</center>| LFW | CFP-FF | CFP-FP | AgeDB-30 | MegaFace
 ---|---|---|---|---|---
<center>LResNet100E-IR</center>|<center>99.77</center>|<center>99.84</center>|<center>98.27</center>|<center>98.25</center>|<center>98.35</center>
<center>LResNet50E-IR</center>|<center>99.80</center>|<center>99.83</center>|<center>92.17</center>|<center>97.70</center>|<center>97.26</center>
<center>LResNet34E-IR</center>|<center>99.67</center>|<center>99.83</center>|<center>90.71</center>|<center>97.63</center>|<center>96.09</center>
<center>MobileFaceNet</center>|<center>99.45</center>|<center>99.49</center>|<center>89.77</center>|<center>95.72</center>|<center>88.63</center>

**2. 基于MS1M-ArcFace训练模型**
<center>模型</center>| LFW | CFP-FF | CFP-FP | AgeDB-30 | MegaFace
 ---|---|---|---|---|---
<center>MobileFaceNet*</center>|<center>99.52<br>99.52</center>|<center>99.44<br>99.50</center>|<center>94.24<br>93.89</center>|<center>96.23<br>96.48</center>|<center>**90.51**<br>90.42</center>
<center>MobileFaceNet-triplet*</center>|<center>99.50</center>|<center>99.44</center>|<center>94.87</center>|<center>96.83</center>|<center>90.10</center>
<center>LResNet100E-IR</center>|<center>99.73</center>|<center>99.85</center>|<center>97.63</center>|<center>97.85</center>|<center></center>
<center>LResNet50E-IR</center>|<center>99.68</center>|<center>99.81</center>|<center>96.89</center>|<center>97.70</center>|<center>96.38</center>
<center>LResNet34E-IR</center>|<center>**99.78**</center>|<center>99.80</center>|<center>**97.19**</center>|<center>**98.02**</center>|<center>**97.36**</center>

**3. VSP模型**
<center>模型</center>| LFW | CFP-FF | CFP-FP | AgeDB-30 | MegaFace
 ---|---|---|---|---|---
<center>LResNet100E-IR-1*</center>|<center>99.82</center>|<center>99.81</center>|<center>98.50</center>|<center>98.12</center>|<center>98.14</center>
<center>LResNet100E-IR-2*</center>|<center>99.82</center>|<center>99.84</center>|<center>94.45</center>|<center>98.03</center>|<center>97.33</center>

## 数据集
- **LFW:** [http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)
- **CFP:** [http://www.cfpw.io/index.html](http://www.cfpw.io/index.html)
- **AgeDB** [https://ibug.doc.ic.ac.uk/resources/agedb/](https://ibug.doc.ic.ac.uk/resources/agedb/)
- **MegaFace:** [http://megaface.cs.washington.edu/](http://megaface.cs.washington.edu/)
- **MS-Celeb-1M:** [https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)
