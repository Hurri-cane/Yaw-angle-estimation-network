# author:Hurricane
# date:  2021/9/07
# E-mail:hurri_cane@qq.com


import argparse
import pickle
import time
from thop import profile
import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from utils.config import select_device

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Yawing_Dataset(Dataset):
	def __init__(self, data_path, device):
		with open(data_path, 'rb')as f:
			_ = pickle.load(f)
		self.data, self.label = _["data"], _["label"]
		self.device = device
		print(len(self.data))

	def __getitem__(self, index):
		data = self.data[index]
		label = self.label[index]
		return data, label

	def __len__(self):
		return len(self.label)


def show_result(epoch_losses, epoch_train_acc, epoch_val_cc, lr_list):
	plt.figure(1)
	plt.plot(epoch_losses, color='r')
	plt.ylim((0, 1))
	plt.grid(True)
	plt.xlabel('epoch')

	plt.ylabel('loss')

	plt.figure(2)
	plt.plot(epoch_train_acc, color='b')
	plt.ylim((0, 30))
	plt.grid(True)
	plt.xlabel('epoch')

	plt.ylabel('train_acc/°')
	plt.show()

	plt.figure(3)
	plt.plot(epoch_val_cc, color='g')
	plt.ylim((0, 30))
	plt.grid(True)
	plt.xlabel('epoch')

	plt.ylabel('val_acc/°')
	plt.show()

	plt.figure(4)
	plt.plot(lr_list, color='violet')
	plt.ylim((0, 0.0011))
	plt.grid(True)
	plt.xlabel('epoch')

	plt.ylabel('Ir')
	plt.show()


class BasicConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1):
		super(BasicConv, self).__init__()

		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False)
		self.bn = nn.BatchNorm2d(out_channels)
		self.activation = nn.LeakyReLU(0.1)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.activation(x)
		return x


class Upsample(nn.Module):
	def __init__(self, in_channels, out_channels, scale_factor):
		super(Upsample, self).__init__()

		self.upsample = nn.Sequential(
			BasicConv(in_channels, out_channels, 1),
			nn.Upsample(scale_factor=scale_factor, mode='nearest')
		)

	def forward(self, x, ):
		x = self.upsample(x)
		return x


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		min_c = 64
		# Convolution by row
		self.conv_for_R = BasicConv(1, min_c, (1, 8))

		self.conv_for_C1 = BasicConv(min_c, min_c, (1, 1))
		self.conv_for_C2 = BasicConv(min_c, min_c * 2, (2, 1))
		self.conv_for_C3 = BasicConv(min_c, min_c, (3, 1))
		self.conv_for_C4 = BasicConv(min_c, min_c * 2, (4, 1))
		self.conv_for_C5 = BasicConv(min_c, min_c * 2, (5, 1))
		self.conv_for_C6 = BasicConv(min_c, min_c * 4, (6, 1))

		self.upsample_for_C4 = Upsample(min_c * 2, min_c * 1, 2)
		self.upsample_for_C5 = Upsample(min_c * 2, min_c * 1, 2)

		self.conv_for_C1_C4 = BasicConv(min_c * 2, min_c * 4, (6, 1))
		self.conv_for_C3_C5 = BasicConv(min_c * 2, min_c * 4, (4, 1))
		self.conv_for_C2_1 = BasicConv(min_c * 2, min_c * 4, (5, 1))

		self.conv_for_C_all = BasicConv(min_c * 16, min_c * 16, (1, 1))

		self.conv_for_all = BasicConv(min_c * 16, min_c * 16, (1, 1))
		self.fconnect = nn.Sequential(
			# nn.Linear(min_c * 16, 2048),
			nn.LeakyReLU(0.1),
			nn.Dropout(0.5),
			nn.Linear(min_c * 16, 2),
		)

	def forward(self, h):
		R = self.conv_for_R(h)

		C1 = self.conv_for_C1(R)
		C2 = self.conv_for_C2(R)
		C3 = self.conv_for_C3(R)
		C4 = self.conv_for_C4(R)
		C5 = self.conv_for_C5(R)
		C6 = self.conv_for_C6(R)

		C4_Upsample = self.upsample_for_C4(C4)[:, :, :, 0].unsqueeze(3)
		C5_Upsample = self.upsample_for_C5(C5)[:, :, :, 0].unsqueeze(3)

		C1_C4 = torch.cat([C1, C4_Upsample], axis=1)
		C3_C5 = torch.cat([C3, C5_Upsample], axis=1)

		C1_C4 = self.conv_for_C1_C4(C1_C4)
		C3_C5 = self.conv_for_C3_C5(C3_C5)
		C2 = self.conv_for_C2_1(C2)

		C_all = torch.cat([C1_C4, C3_C5, C2, C6], axis=1)
		C_all = self.conv_for_C_all(C_all)

		out = C_all
		out = self.conv_for_all(out)[:, :, 0, 0]
		out = self.fconnect(out)

		return out


def squared_loss(y_hat, y):
	diff = abs(y_hat - y.view(y_hat.size()))
	# t1 = time.time()
	angle = diff[:, 0]
	single = diff[:, 1]
	angle_bigger_1 = angle * (angle >= 1)
	angle_smaller_1 = angle * (angle < 1)
	loss_single = sum(single ** 3 / 3)
	loss_angle = sum((angle_bigger_1 - 0.5) * (angle >= 1)) + sum(angle_smaller_1 ** 2 / 2)
	loss = loss_single + loss_angle
	# print("time:", time.time() - t1, "loss:", loss)
	# print("epoch:", epoch, loss.item())
	return loss


def calculate_degree(data):
	# convert single + angle to Yaw
	angle = data[:, 0]
	single = data[:, 1]
	angle_bigger = angle * 180 * (single >= 0.5)
	angle_smaller = (360 - angle * 180) * (single < 0.5)
	out = angle_bigger + angle_smaller
	# adjust Yaw
	out_correct = out * ((out >= 0) & (out <= 360))
	out_bigger_360 = (out - 360) * (out > 360)
	out_smaller_0 = (out + 360) * (out < 0)
	res = out_correct + out_bigger_360 + out_smaller_0
	return res


def calculate_acc(pre, l):
	diff = abs(pre - l)
	correct_pre = (pre <= 360) & (pre > 0)
	correct_diff_in_180_360 = (diff > 180) & (diff < 360) & correct_pre
	error_diff = ~ correct_diff_in_180_360
	res = (sum(diff * error_diff) + sum((360 - diff) * correct_diff_in_180_360)) / len(l)
	return res


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
	parser.add_argument('--workers', type=int, default=0, help='num of DataLoader workers, i.e. 0 or 4 or 8')
	parser.add_argument('--epochs', type=int, default=1000)
	opt = parser.parse_args()
	main_path = r"./Yaw_angle_dataset"

	train_path = os.path.join(main_path, "train/yawing.data")
	val_path = os.path.join(main_path, "val/yawing.data")
	weight_path = os.path.join(main_path, "graph_weight")

	train_time = datetime.datetime.now().strftime('%Y-%m-%d-%Hh%Mm%Ss')

	device = select_device(opt.device)
	batch_size = opt.batch_size
	num_workers = opt.workers
	epochs = opt.epochs
	# print(type(num_workers))
	train_Dataset = Yawing_Dataset(data_path=train_path, device=device)
	train_loader = DataLoader(dataset=train_Dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	val_Dataset = Yawing_Dataset(data_path=val_path, device=device)
	val_loader = DataLoader(dataset=val_Dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	Ir = 1e-3
	momentum = 0.937
	# model = Model(in_dim=4, hidden_dim=1000, out_dim=1).to(device)
	model = Model().to(device)
	optim = torch.optim.Adam(model.parameters(), lr=Ir, betas=(momentum, 0.999))

	# 动态调整学习率
	scheduler = MultiStepLR(optim, milestones=[100, 200], gamma=0.8)

	epoch_Losses = []
	epoch_Accuracy = []
	epoch_val_Accuracy = []
	lr_list = []
	time_strat = time.time()
	accuracy_min = 1e7
	for epoch in range(epochs):
		losses = []
		Accuracy = []
		val_Accuracy = []
		model.train()
		for batch, labels in train_loader:
			# Read the data from train_loader
			with torch.no_grad():
				batch = batch.to(device)
				labels = labels.to(device)

			optim.zero_grad()
			logits = model(batch)
			# loss = F.cross_entropy(logits, labels)
			loss = squared_loss(logits, labels)

			loss.backward()
			optim.step()
			degree_logits = calculate_degree(logits)
			degree_label = calculate_degree(labels)
			accuracy = calculate_acc(degree_logits, degree_label)

			losses.append(loss.detach().to("cpu").numpy())
			Accuracy.append(accuracy.detach().to("cpu").numpy())

			print("", epoch, "T:", degree_label[0:5].detach().to("cpu").numpy().tolist(),
				  '\n', epoch, "P:", degree_logits[0:5].detach().to("cpu").numpy().tolist())

		epoch_accuracy = sum(Accuracy) / len(Accuracy)

		epoch_Losses.append(sum(losses) / len(losses))
		epoch_Accuracy.append(sum(Accuracy) / len(Accuracy))

		# val
		model.eval()

		for batch, labels in val_loader:
			batch = batch.to(device)
			labels = labels.to(device)

			val_logits = model(batch)

			degree_logits = calculate_degree(val_logits)
			degree_label = calculate_degree(labels)
			val_accuracy = calculate_acc(degree_logits, degree_label)

			val_Accuracy.append(val_accuracy.detach().to("cpu").numpy())

		epoch_val_accuracy = sum(val_Accuracy) / len(val_Accuracy)
		if epoch_val_accuracy < accuracy_min:
			if accuracy_min != 1e7:
				file_rm_name = train_time + "-Accuracy %f.pt" % accuracy_min
				model_rm_path = os.path.join(weight_path, file_rm_name)
				os.remove(model_rm_path)

			accuracy_min = epoch_val_accuracy
			file_s_name = train_time + "-Accuracy %f.pt" % accuracy_min
			model_s_path = os.path.join(weight_path, file_s_name)
			torch.save(model, model_s_path)

		epoch_val_Accuracy.append(epoch_val_accuracy)
		lr_list.append(optim.param_groups[0]['lr'])
		scheduler.step()

	inputs = torch.randn(1, 1, 6, 8).to(device)
	flops, params = profile(model, (inputs,))
	print('flops: ', flops, 'params: ', params)

	print("Time cost:%s h" % ((time.time() - time_strat) / 3600))
	print("Epoch:", epoch_Losses.index(min(epoch_Losses)), "min_losses:", min(epoch_Losses))
	print("Epoch:", epoch_Accuracy.index(min(epoch_Accuracy)), "min_train_Accuracy:", min(epoch_Accuracy))
	print("Epoch:", epoch_val_Accuracy.index(min(epoch_val_Accuracy)), "min_val_Accuracy:", min(epoch_val_Accuracy))

	show_result(epoch_Losses, epoch_Accuracy, epoch_val_Accuracy, lr_list)
	print("ALL Done！")
