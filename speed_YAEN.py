# author:Hurricane
# date:  2021/11/22
# E-mail:hurri_cane@qq.com

import torch
import time
import numpy as np
import cv2
import random
from models.experimental import attempt_load
from utils.config import select_device, increment_path, set_logging, check_img_size, non_max_suppression
from utils.transport import car2full_img, adjust_box, svae_xlsx
from torch.utils.data import Dataset, DataLoader
from models.baselines import Model as Model
from models.baselines import BasicConv, Upsample


# torch.backends.cudnn.deterministic = False
def get_object_model(weights, device):
	imgsz = 640
	device = select_device(device)
	# Initialize
	set_logging()
	half = device.type != 'cpu'  # half precision only supported on CUDA
	# Load model
	model = attempt_load(weights, map_location=device)  # load FP32 model
	stride = int(model.stride.max())  # model stride
	imgsz = check_img_size(imgsz, s=stride)  # check img_size
	if half:
		model.half()  # to FP16
	# Run inference
	if device.type != 'cpu':
		model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
	return model


def get_head_model(weight_path, device):
	model = torch.load(weight_path, map_location=device)
	return model


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


def analyze_bbox(bbox):
	objects = []
	bbox = np.array(bbox)
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
	return res


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


def calculate(img0, object_model, head_model, device):
	Head_angle_in_one_img = []
	img, scale0, dw0, dh0 = tr_img(img0, device, half, 640)
	# vehicle inspection
	pred = object_model(img, augment=False)[0]
	# Apply NMS
	pred_object = non_max_suppression(pred, 0.5, 0.45, agnostic=False)[0]
	object_imgs = []
	object_part_positions = []
	for i, det in enumerate(pred_object):  # detections per image
		bbox = det[:4]
		bbox = adjust_box(bbox, scale0, dw0, dh0)
		det = np.array(det.to("cpu"))
		det[:4] = bbox
		part_img0 = img0[bbox[1]:bbox[3], bbox[0]:bbox[2]]
		object_part_positions.append(det)
	objects = analyze_bbox(object_part_positions)
	for obj in objects:
		if len(obj) > 2:
			head = Encode(img0.shape, obj, "one_hot_add_single", device)
			head = head.unsqueeze(0)
			pred = head_model(head)
			head_angle = calculate_degree(pred, "one_hot_add_single").item()
			Head_angle_in_one_img.append(head_angle)
	return Head_angle_in_one_img


if __name__ == '__main__':

	torch.backends.benchmark = True
	car_weight_path = "./weights/car_car_part_model.pt"
	HADN_weight_path = "./weights/yaw_angle_estimation.pt"
	device = ''
	if torch.cuda.is_available():
		device = 'cuda:0'
	else:
		device = 'cpu'

	object_model = get_object_model(car_weight_path, device)
	head_model = get_head_model(HADN_weight_path, device)
	object_model.eval()
	head_model.eval()
	img0 = cv2.imread("data/speed_test.jpg")
	half = True
	for i in range(10):
		res = calculate(img0, object_model, head_model, device)

	t_all = []
	for i in range(100):
		t1 = time.time()
		res = calculate(img0, object_model, head_model, device)
		t2 = time.time()
		t_all.append(t2 - t1)

	print('average time:', np.mean(t_all) / 1)
	print('average fps:', 1 / np.mean(t_all))

	print('fastest time:', min(t_all) / 1)
	print('fastest fps:', 1 / min(t_all))

	print('slowest time:', max(t_all) / 1)
	print('slowest fps:', 1 / max(t_all))

	print("All Done")
