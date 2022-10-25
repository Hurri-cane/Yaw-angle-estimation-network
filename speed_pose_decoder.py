# author:Hurricane
# date:  2021/11/22
# E-mail:hurri_cane@qq.com

import torch
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.baselines import Model as Model
from models.baselines import BasicConv, Upsample
import os

# torch.backends.cudnn.deterministic = False

torch.backends.benchmark = True
weight_path = "./weights/yaw_angle_estimation.pt"
device = ''
if torch.cuda.is_available():
	device = 'cuda:0'
else:
	device = 'cpu'
model = torch.load(weight_path, map_location=device)
model.eval()

x = torch.zeros((1, 1, 6, 8)).cuda() + 1
for i in range(10):
	y = model(x)

t_all = []
for i in range(100):
	t1 = time.time()
	y = model(x)
	t2 = time.time()
	t_all.append(t2 - t1)

print('average time:', np.mean(t_all) / 1)
print('average fps:', 1 / np.mean(t_all))

print('fastest time:', min(t_all) / 1)
print('fastest fps:', 1 / min(t_all))

print('slowest time:', max(t_all) / 1)
print('slowest fps:', 1 / max(t_all))

print("All Done")
