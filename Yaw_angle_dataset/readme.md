# Data source: Team of Professor Bin Zou, School of Automotive Engineering, Wuhan University of Technology

# Contact email: hurri_cane@qq.com

# Data collection platform:

## Observation vehicle equipment:

Velodyne VLP-32C * 1；FLIR GS3-U3 * 2； OXTSGPS RT3000 v2 * 1

![主车设备](https://user-images.githubusercontent.com/32425429/147375598-b4ce75c6-3ca4-4fd2-8561-a21f986e60bd.jpg)

## Observed vehicle equipment:

OXTSGPS RT3000 v2 * 1

<img src=https://user-images.githubusercontent.com/32425429/147375599-c931208e-eb87-4409-8be6-526958c958a5.jpg width=40% />


## Auxiliary equipment.

OXTS RT-BASE
<img src="https://user-images.githubusercontent.com/32425429/147375600-a8db0d54-5012-4fe7-8f54-4016613625ef.png" style="zoom:25%;" />



# Document Description:

graph_weight ：The folder where training results are stored

org ：original data

train ：The Data for the training model

val ：The data used to test the model

raw_data_example：Example of raw data collected by the above device, note that only part of the raw data is shown here, if you need more raw data, please refer to this link [GoogleDrive](https://drive.google.com/drive/folders/17uoB-aNu3g1SA42K_FgYyvCALzejKvUE)/[BaiduDrive(code:2233 )](https://pan.baidu.com/s/1oBU9uWupAsT147W8kIJyEA?pwd=2233 )

the yawing.data in train/val folder contains 14277/1586 sample of mappings between transverse pendulum angles and part combinations, which can be extracted using Python's pickle tool, for example

```
import pickle
with open(data_path, 'rb')as f:
    data = pickle.load(f)
```


For more details, please refer to the [train.py](../train.py)  file

# 

