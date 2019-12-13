<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [**相关算法**](#相关算法)
	- [**人脸检测**](#人脸检测)
		- [RetinaFace](#retinaface)
	- [**人脸对齐**](#人脸对齐)
		- [Dense U-Net](#dense-u-net)
	- [**人脸识别**](#人脸识别)
		- [ArcFace](#arcface)
	- [参考文献](#参考文献)

<!-- /TOC -->

# **相关算法**

## **人脸检测**

### RetinaFace

## **人脸对齐**

### Dense U-Net

## **人脸识别**

### ArcFace

&emsp;&emsp;根据论文，文章首先介绍了三种利用卷积神经网络识别人脸的主要属性。先是训练数据，介绍了主要的人脸识别训练数据集；其次是网络结构，介绍了各种卷积神经网络；第三是损失函数，介绍了基于欧几里得距离的损失函数和基于角度和余弦的损失函数。<br>
&emsp;&emsp;文章介绍了从SoftMax到ArcFace损失函数。介绍了：(1)SoftMax损失函数；(2)权重归一化；(3)Angular Margin倍数损失函数；(4)特征归一化；(5)Cosine Margin损失函数；(6)Angular Margin损失函数；

## 参考文献
[1] **ArcFace:** Additive Angular Margin Loss for Deep Face Recognition, Jiankang Deng, Jia Guo, Niannan Xue,
Stefanos Zafeiriou, [https://arxiv.org/abs/1801.07698](https://arxiv.org/abs/1801.07698)<br>
[2] **RetinaFace:** Single-stage Dense Face Localisation in the Wild, Jiankang Deng, Jia Guo, Yuxiang Zhou,
Jinke Yu, Irene Kotsia, Stefanos Zafeiriou, [https://arxiv.org/abs/1905.00641](https://arxiv.org/abs/1905.00641)<br>
[3] Stacked **Dense U-Nets** with Dual Transformers for Robust Face Alignment, Jia Guo, Jiankang Deng,
Niannan Xue, Stefanos Zafeiriou, [https://arxiv.org/abs/1812.01936](https://arxiv.org/abs/1812.01936)<br>
