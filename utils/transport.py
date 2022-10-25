# author:Hurricane
# date:  2021/10/7
# E-mail:hurri_cane@qq.com
import cv2
import math
import pandas as pd
import torch
import numpy as np
import random


def node_process(h, node_num):
	if len(h) > node_num:
		random_indx = random.sample(range(0, len(h)), node_num)
		new_h = h[random_indx]
	else:
		add_data = torch.zeros(node_num - len(h), h.shape[1])
		new_h = torch.cat((h, add_data), dim=0)
	# Disorder
	shuffle_index = torch.randperm(new_h.size(0))
	new_h = new_h[shuffle_index]
	return new_h


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
	# Resize and pad image while meeting stride-multiple constraints
	shape = img.shape[:2]  # current shape [height, width]
	if isinstance(new_shape, int):
		new_shape = (new_shape, new_shape)

	# Scale ratio (new / old)
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
	if not scaleup:  # only scale down, do not scale up (for better test mAP)
		r = min(r, 1.0)

	# Compute padding
	ratio = r, r  # width, height ratios
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
	if auto:  # minimum rectangle
		dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
	elif scaleFill:  # stretch
		dw, dh = 0.0, 0.0
		new_unpad = (new_shape[1], new_shape[0])
		ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

	dw /= 2  # divide padding into 2 sides
	dh /= 2

	if shape[::-1] != new_unpad:  # resize
		img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
	return img, ratio, dw, dh


def tr_img(img0, device, half, img_size):
	img, ratio, dw, dh = letterbox(img0, img_size)
	assert ratio[0] == ratio[1]
	img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
	img = np.ascontiguousarray(img)
	img = torch.from_numpy(img).to(device)
	img = img.half() if half else img.float()  # uint8 to fp16/32
	img /= 255.0  # 0 - 255 to 0.0 - 1.0
	if img.ndimension() == 3:
		img = img.unsqueeze(0)
	return img, ratio, dw, dh


def average_a_proportions(a_proportions):
	max_a = max(a_proportions)
	res = list(map(lambda x: x / max_a, a_proportions))
	return res


def Encode(img_shape, obj, data_model, device):
	car = obj[obj[:, 5] == 0]
	car_parts = obj[obj[:, 5] != 0]
	Area = img_shape[0] * img_shape[1]
	categorys = []
	centers_x = []
	centers_y = []
	a_proportions = []
	for pos in car_parts:
		box = pos[:4]
		new_box = box
		center_x = (new_box[0] + new_box[2]) / 2 / img_shape[1]
		center_y = (new_box[1] + new_box[3]) / 2 / img_shape[0]
		are = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
		a_proportion = (are / Area)
		# Adding one here is to add 0 nodes, and subtracting one is to remove the impact of vehicle categories
		# cls_id = int(pos[-1] + 1 - 1)
		cls_id = int(pos[-1])

		categorys.append(cls_id)
		centers_x.append(center_x)
		centers_y.append(center_y)
		a_proportions.append(a_proportion)
	a_proportions = average_a_proportions(a_proportions)
	if "one_hot" in data_model:
		One_hot = np.zeros((1, 5))
		for c in categorys:
			one_hot = np.zeros((1, 5))

			one_hot[0, c] = 1
			# Remove the boot array
			One_hot = np.concatenate((One_hot, one_hot))

		One_hot = One_hot[1:]
		feats = list(
			zip(One_hot[:, 0], One_hot[:, 1], One_hot[:, 2], One_hot[:, 3], One_hot[:, 4], centers_x, centers_y,
				a_proportions))
		data = torch.Tensor(feats)
		data = node_process(data, 6)
		data = torch.unsqueeze(data, 0)
		return data.to(device)
	else:
		feats = list(zip(categorys, centers_x, centers_y, a_proportions))
		data = torch.Tensor(feats)
		data = node_process(data, 6)
		data = torch.unsqueeze(data, 0)
		return data.to(device)


def car2full_img(car_box, car_part):
	new_box = []
	for pos in car_part:
		box = [pos[0] + car_box[0], pos[1] + car_box[1], pos[2] + car_box[0], pos[3] + car_box[1]]
		box.extend(pos[4:])
		new_box.append(box)
	return new_box


def adjust_box(box, scale, dw, dh):
	dw0 = math.floor(dw)  	# Supplementary gray block size on the left
	dw1 = math.ceil(dw)  	# Supplementary gray block size on the right
	dh0 = math.floor(dh) 	# Supplementary gray block size on the top
	dh1 = math.ceil(dh) 	# Supplementary gray block size on the bottom
	car_bbox = np.array(box.to("cpu"))
	# 去灰边之后的Bbox
	car_bbox = [car_bbox[0] - dw0, car_bbox[1] - dh0, car_bbox[2] - dw0, car_bbox[3] - dh0]
	car_bbox = np.array(car_bbox) / scale[0]
	car_bbox = list(map(int, car_bbox))
	new_box = list(map(lambda x: 0 if x < 0 else x, car_bbox))
	return new_box


def svae_xlsx(path, name, Head_angle):
	name = list(map(lambda x: x.split("-171")[0], name))
	Head_angle = list(map(lambda x: x[0] if len(x) == 1 else x, Head_angle))
	data = {"Timestamp": name, "Pre": Head_angle}
	df = pd.DataFrame(data)
	print(df)
	df.to_excel(path)


def analyze_bbox(bbox):
	objects = []
	bbox = np.array(bbox)
	if len(bbox) > 1:
		cars = bbox[bbox[:, 5] == 0]
		parts = bbox[bbox[:, 5] != 0]
		parts_center = np.array(((parts[:, 0] + parts[:, 2]), (parts[:, 1] + parts[:, 3]))).T / 2
		for car in cars:
			discriminate1 = parts_center[:, 0] >= car[0]
			discriminate2 = parts_center[:, 0] <= car[2]
			discriminate3 = parts_center[:, 1] >= car[1]
			discriminate4 = parts_center[:, 1] <= car[3]
			Discriminate = discriminate1 & discriminate2 & discriminate3 & discriminate4
			correct_parts = parts[Discriminate]
			one_objects = np.concatenate((car[None], correct_parts))
			objects.append(one_objects)
		res = objects
	else:
		res = objects
	return res


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

