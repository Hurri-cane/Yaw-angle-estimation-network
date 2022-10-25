# author:Hurricane
# date:  2021/9/07
# E-mail:hurri_cane@qq.com


import pickle
import time
import pandas as pd
import datetime
import torch
import os
from torch.utils.data import Dataset, DataLoader
from models.baselines import Model as Model
from models.baselines import BasicConv, Upsample


# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


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


def calculate_degree(data, model):
	if "single" in model:
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

	else:
		out = data.squeeze(1) * 360
		out_correct = out * ((out >= 0) & (out <= 360))
		out_bigger_360 = (out - 360) * (out > 360)
		out_smaller_0 = (out + 360) * (out < 0)
		res = out_correct + out_bigger_360 + out_smaller_0
		return res


def analyse(label, pre, path):
	assert len(label) == len(pre)
	data = {"Label": label.detach().to("cpu").numpy(), "Pre": pre.detach().to("cpu").numpy()}
	df = pd.DataFrame(data)
	# print(df)
	df.to_excel(path)


def calculate_acc(pre, l):
	diff = abs(pre - l)
	# t1 = time.time()
	diff_in_180_360 = (diff > 180) & (diff < 360)
	another_diff = ~ diff_in_180_360
	AP = (sum(diff * another_diff) + sum((360 - diff) * diff_in_180_360)) / len(l)
	GP_5 = sum(diff <= 5)
	GP_10 = sum(diff <= 10)
	res = (AP, GP_5 / len(pre) * 100, GP_10 / len(pre) * 100)
	# print(time.time() - t1, res)
	return res


if __name__ == '__main__':
	main_path = r'./Yaw_angle_dataset'
	weight_path = r"./weights"
	weight_name = r"yaw_angle_estimation.pt"
	data_model = "one_hot_add_single"
	data_path = os.path.join(main_path, r"val/yawing.data")

	xlsx_path = os.path.join(r"./analyse", weight_name.split(".pt")[0] + ".xlsx")
	train_time = datetime.datetime.now().strftime('%Y-%m-%d-%Hh%Mm%Ss')

	device = ''
	if torch.cuda.is_available():
		device = 'cuda:0'
	else:
		device = 'cpu'
	train_Dataset = Yawing_Dataset(data_path=data_path, device=device)
	train_loader = DataLoader(dataset=train_Dataset, batch_size=128, shuffle=True, num_workers=0)

	# model = Model(in_dim=4, hidden_dim=1000, out_dim=1).to(device)

	epoch_Accuracy = []
	time_strat = time.time()
	accuracy_min = 1e7
	model = torch.load(os.path.join(weight_path, weight_name), map_location=device)
	model.eval()
	losses = []
	Accuracy = []
	Label = torch.tensor([0]).to('cuda:0')
	Pre = torch.tensor([0]).to('cuda:0')
	for batch, labels in train_loader:
		# Read the data from train_loader
		batch = batch.to(device)
		labels = labels.to(device)

		logits = model(batch)

		# save_result
		degree_logits = calculate_degree(logits, data_model)
		degree_label = calculate_degree(labels, data_model)

		Pre = torch.cat((Pre, degree_logits))
		Label = torch.cat((Label, degree_label))
	Pre = Pre[1:]
	Label = Label[1:]
	AP, GP_5, GP_10 = calculate_acc(Pre, Label)

	analyse(Label, Pre, xlsx_path)

	print(weight_name)
	print("E: %0.2f" % AP)
	print("EP_5: %0.2f" % GP_5)
	print("EP_10: %0.2f" % GP_10)
	print("Time cost:%s s" % ((time.time() - time_strat)))
	print("ALL Done！")
