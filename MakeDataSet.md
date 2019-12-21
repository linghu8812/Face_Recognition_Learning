<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [制作insightface record数据集](#制作insightface-record数据集)
	- [文件说明](#文件说明)
	- [处理流程](#处理流程)

<!-- /TOC -->

# 制作insightface record数据集

## 文件说明
文件名|说明
---|---
[parsing_record.py](parsing_record.py)|解析face_emore.rec文件
[make_vsp_dataset.py](make_vsp_dataset.py)|对自己的数据集进行人脸对齐及校正
[clean_dataset.py](clean_dataset.py)|清洗人脸数据集
[same_class_predict.py](same_class_predict.py)|判断两个文件夹下的图片是否为同一类

## 处理流程
1. 对于需要合并数据集的record文件，使用[parsing_record.py](parsing_record.py)文件，将record文件转换为图片文件；
2. 对于未进行人脸检测的数据，使用[make_vsp_dataset.py](make_vsp_dataset.py)文件进行人脸检测，并将人脸检测结果保存为112x112大小的图片；
3. 使用[clean_dataset.py](clean_dataset.py)文件判断文件中的异常数据，并需要进行人工确认；
4. 使用[same_class_predict.py](same_class_predict.py)判断两个文件夹下的图片是否为同一类；
