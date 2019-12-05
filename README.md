<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [**InsightFace算法学习**](#insightface算法学习)
	- [开源仓库](#开源仓库)
		- [识别算法配置](#识别算法配置)
	- [算法测试](#算法测试)
	- [相关算法](#相关算法)
	- [数据集](#数据集)
	- [参考文献](#参考文献)

<!-- /TOC -->

# **InsightFace算法学习**

## 开源仓库
- **InsightFace:** [https://github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)<br>

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

## 算法测试
1. 直接训练(MS1MV2-Arcface)数据集，基于MobileFaceNet网络和ArcFace损失函数进行训练。
````
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network y1 --loss arcface --dataset emore
````
训练配置
````
Namespace(batch_size=256, ckpt=3, ctx_num=1, dataset='emore', frequent=20, image_channel=3, kvstore='device', loss='arcface', lr=0.1, lr_steps='100000,160000,220000', models_root='./models', mom=0.9, network='y1', per_batch_size=256, pretrained='', pretrained_epoch=1, rescale_threshold=0, verbose=2000, wd=0.0005) {'loss_m1': 1.0, 'loss_m2': 0.5, 'loss_m3': 0.0, 'net_act': 'prelu', 'emb_size': 128, 'data_rand_mirror': True, 'loss_name': 'margin_softmax', 'val_targets': ['lfw', 'cfp_fp', 'agedb_30'], 'ce_loss': True, 'net_input': 1, 'image_shape': [112, 112, 3], 'net_blocks': [1, 4, 6, 2], 'fc7_lr_mult': 1.0, 'ckpt_embedding': True, 'net_unit': 3, 'net_output': 'GDC', 'count_flops': True, 'num_workers': 1, 'batch_size': 256, 'memonger': False, 'data_images_filter': 0, 'dataset': 'emore', 'num_classes': 85742, 'fc7_no_bias': False, 'loss': 'arcface', 'data_color': 0, 'loss_s': 64.0, 'dataset_path': '../datasets/faces_emore', 'data_cutoff': False, 'net_se': 0, 'net_multiplier': 1.0, 'fc7_wd_mult': 1.0, 'network': 'y1', 'per_batch_size': 256, 'net_name': 'fmobilefacenet', 'workspace': 256, 'max_steps': 0, 'bn_mom': 0.9}
('in_network', {'loss_m1': 1.0, 'loss_m2': 0.5, 'loss_m3': 0.0, 'net_act': 'prelu', 'emb_size': 128, 'data_rand_mirror': True, 'loss_name': 'margin_softmax', 'val_targets': ['lfw', 'cfp_fp', 'agedb_30'], 'ce_loss': True, 'net_input': 1, 'image_shape': [112, 112, 3], 'net_blocks': [1, 4, 6, 2], 'fc7_lr_mult': 1.0, 'ckpt_embedding': True, 'net_unit': 3, 'net_output': 'GDC', 'count_flops': True, 'num_workers': 1, 'batch_size': 256, 'memonger': False, 'data_images_filter': 0, 'dataset': 'emore', 'num_classes': 85742, 'fc7_no_bias': False, 'loss': 'arcface', 'data_color': 0, 'loss_s': 64.0, 'dataset_path': '../datasets/faces_emore', 'data_cutoff': False, 'net_se': 0, 'net_multiplier': 1.0, 'fc7_wd_mult': 1.0, 'network': 'y1', 'per_batch_size': 256, 'net_name': 'fmobilefacenet', 'workspace': 256, 'max_steps': 0, 'bn_mom': 0.9})
````
训练结果
````

````



## 相关算法
- **人脸检测：** RetinaFace<br>
- **人脸对齐：** Dense U-Net<br>
- **人脸识别：** ArcFace<br>

## 数据集
- **LFW:** [http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)
- **CFP:** [http://www.cfpw.io/index.html](http://www.cfpw.io/index.html)
- **MegaFace:** [http://megaface.cs.washington.edu/](http://megaface.cs.washington.edu/)
- **MS-Celeb-1M:** [https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)


## 参考文献
[1] **ArcFace:** Additive Angular Margin Loss for Deep Face Recognition, Jiankang Deng, Jia Guo, Niannan Xue,
Stefanos Zafeiriou, [https://arxiv.org/abs/1801.07698](https://arxiv.org/abs/1801.07698)<br>
[2] **RetinaFace:** Single-stage Dense Face Localisation in the Wild, Jiankang Deng, Jia Guo, Yuxiang Zhou,
Jinke Yu, Irene Kotsia, Stefanos Zafeiriou, [https://arxiv.org/abs/1905.00641](https://arxiv.org/abs/1905.00641)<br>
[3] Stacked **Dense U-Nets** with Dual Transformers for Robust Face Alignment, Jia Guo, Jiankang Deng,
Niannan Xue, Stefanos Zafeiriou, [https://arxiv.org/abs/1812.01936](https://arxiv.org/abs/1812.01936)<br>
