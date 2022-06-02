# Yaw-angle-estimation-network
![image](https://user-images.githubusercontent.com/32425429/147311231-a7fb71f2-2e8c-489d-856a-5a1666f95b2f.png)

## Introduction

 In this work, we proposed a framework for accurately predicting the object's pose based on the arrangement of parts. We apply this framework to estimate the yaw angle of vehicles, and we call it YAEN. YAEN can estimate the yaw angle of vehicles using a monocular camera. 

Demo video is available [You Tube](https://youtu.be/TfCqXiFiCEY) or [Bilibili](https://www.bilibili.com/video/BV1sS4y1M7Aq/)  .  

![image-20211224114423565](https://user-images.githubusercontent.com/32425429/147342836-b5211eb8-c842-4600-b42b-75ff979dcc30.png)
## Install

Please see [INSTALL.md](./INSTALL.md)

## Train

For Pose Decoding Network training, run

```python
python train.py
```

The results of the training model will appear in this folder ["./Yaw_angle_dataset/graph_weight"](./Yaw_angle_dataset/graph_weight).

***

Besides config style settings, we also support command line style one. You can override a setting like

```
python train.py --device 0 --epochs 300
```

The ```epochs``` will be set to 300 during training.

### Trained models

We provide two trained models in  ["./weights"](./weights).

```car_car_part_model``` is the model of the Parts Encoding Network

```yaw_angle_estimation``` is the model of the Pose Decoding Network

## Detect

Yaw-angle-estimation-network can estimate the vehicle's yaw angle from a image. Before we can begin detecting, we need to prepare the predictive object. We’ve provided  some images with the yaw angle annotated.

these images are available   [GoogleDrive](https://drive.google.com/drive/folders/1-noXowdV_pe9VFiJkOhG6brbc8V_LwV2?usp=sharing)/[BaiduDrive(code:2233 )](https://pan.baidu.com/s/1nSSR-jJKwvj-kUT0NJyaMw )

Store the downloaded images in this directory: ["./data/data_val"](./data/data_val).

Run test script

```
python detect.py
```

Testing results are saved in [./results](./results) by default.

## Analyse

We provide the code to analyse the pose decoding network

```
python analyse.py
```

Testing results are saved in [./analyse](./analyse) by default.

## Speed test

To test the runtime, please run

```Shell
python speed_pose_decoder.py
# This will test the speed of pose decoding network

python speed_YAEN.py
# This will test the speed of Yaw-angle-estimation-network
```

It will loop 100 times and calculate the average runtime and fps in your environment.

## Acknowledgment

Zou Bin, Li Wenbo, Tang Luqi, Zhu Xiaoming have made great contributions to this work. Thanks for their helpful contribution.
