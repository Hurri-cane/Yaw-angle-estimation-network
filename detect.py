# author:Hurricane
# date:  2021/10/5
# E-mail:hurri_cane@qq.com


import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import random
import numpy as np
from models.experimental import attempt_load
from utils.config import select_device, increment_path, set_logging, check_img_size, non_max_suppression
from utils.transport import tr_img, adjust_box, svae_xlsx, Encode, analyze_bbox,calculate_degree
from models.baselines import Model
from models.baselines import BasicConv, Upsample


def plot_box(bbox, img, yaw_angle, name, save_dir, label, colors=None, line_thickness=6):
	cars = bbox[bbox[:, 5] == 0][0]
	parts = bbox[bbox[:, 5] != 0]

	# Plots car in img
	tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
	tf = max(tl - 1, 1)  # font thickness

	color_car = colors[-1] or [random.randint(0, 255) for _ in range(3)]
	tl_car = 10
	label_car = "Car" + ",%.2f degree" % yaw_angle
	c1, c2 = (int(cars[0]), int(cars[1])), (int(cars[2]), int(cars[3]))
	cv2.rectangle(img, c1, c2, color_car, thickness=tl_car, lineType=cv2.LINE_AA)
	t_size = cv2.getTextSize(label_car, 0, fontScale=tl_car / 3, thickness=tl_car - 1)[0]
	c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
	cv2.rectangle(img, c1, c2, color_car, -1, cv2.LINE_AA)  # filled
	cv2.putText(img, label_car, (c1[0], c1[1] - 2), 0, tl_car / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

	# Plots car parts in img
	for part in parts:
		c1, c2 = (int(part[0]), int(part[1])), (int(part[2]), int(part[3]))
		color_part = colors[int(part[-1]) - 1]
		cv2.rectangle(img, c1, c2, color_part, thickness=tl, lineType=cv2.LINE_AA)
		label_part = label[int(part[-1])]
		t_size = cv2.getTextSize(label_part, 0, fontScale=tl / 3, thickness=tf)[0]
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.rectangle(img, c1, c2, color_part, -1, cv2.LINE_AA)  # filled
		cv2.putText(img, label_part, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

	# Directories

	if opt.view_img:
		cv2.imshow("img", img)
		cv2.waitKey(0)


def get_object_model(opt):
	weights, imgsz = opt.weights_object, opt.img_size_object
	device = select_device(opt.device)
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


def get_yaw_model(opt, weight_path):
	device = select_device(opt.device)
	model = torch.load(weight_path, map_location=device)
	return model



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights-object', nargs='+', type=str, default='./weights/car_car_part_model.pt',
						help='model.pt path(s)')
	parser.add_argument('--weights-yaw', nargs='+', type=str,
						default='./weights/yaw_angle_estimation.pt',
						help='model.pt path(s)')
	parser.add_argument('--source', type=str, default=r'./data/data_val', help='source')
	parser.add_argument('--img-size-object', type=int, default=640, help='inference size (pixels)')
	parser.add_argument('--conf-thres-object', type=float, default=0.5, help='object confidence threshold')
	parser.add_argument('--iou-thres-object', type=float, default=0.45, help='IOU threshold for NMS')
	parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--view-img', action='store_true', default=False, help='display results')
	parser.add_argument('--save-img', action='store_true', default=True, help='save results to *.jpg')
	parser.add_argument('--save-xlsx', action='store_true', default=True, help='save results to *.xlsx')
	parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
	parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='augmented inference')
	parser.add_argument('--project', default='results', help='save results to project/name')
	parser.add_argument('--name', default='exp', help='save results to project/name')
	opt = parser.parse_args()
	print(opt)
	device = select_device(opt.device)
	half = device.type != 'cpu'  # half precision only supported on CUDA
	data_list = os.listdir(opt.source)
	weight_path = opt.weights_yaw
	data_model = "one_hot_add_single"

	# Generate the folder location where the test results are stored
	save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=False))  # increment run
	(save_dir / 'labels' if opt.save_xlsx else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
	colors = [[237, 46, 81], [73, 104, 45], [84, 234, 31], [97, 38, 69], [170, 178, 32]]

	with torch.no_grad():
		# Create three detection models
		object_model = get_object_model(opt)
		yaw_model = get_yaw_model(opt, weight_path)
		object_model.eval()
		yaw_model.eval()
		time_start = time.time()
		names = object_model.names
		Yaw_angle = []
		for img_num, file_name in enumerate(data_list):
			Yaw_angle_in_one_img = []
			path = os.path.join(opt.source, file_name)
			img0 = cv2.imread(path)
			img, scale0, dw0, dh0 = tr_img(img0, device, half, opt.img_size_object)
			# vehicle inspection
			pred = object_model(img, augment=opt.augment)[0]
			# Apply NMS
			pred_object = \
				non_max_suppression(pred, opt.conf_thres_object, opt.iou_thres_object, agnostic=opt.agnostic_nms)[0]
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
					yaw = Encode(img0.shape, obj, data_model, device)
					yaw = yaw.unsqueeze(0)
					pred = yaw_model(yaw)
					yaw_angle = calculate_degree(pred, data_model).item()

					# Display vehicle detection results and yaw angle detection results on the original image
					plot_box(obj, img0, yaw_angle, file_name, save_dir, names, colors=colors)
					print(len(data_list) - img_num, file_name)
					Yaw_angle_in_one_img.append(yaw_angle)
			if opt.save_img:
				save_path = str(save_dir / file_name)  # img.jpg
				cv2.imwrite(save_path, img0)
			Yaw_angle.append(Yaw_angle_in_one_img)
	if opt.save_xlsx:
		save_name = os.path.join(save_dir, "labels\labels.xlsx")
		svae_xlsx(save_name, data_list, Yaw_angle)
	print("%d imgs cost %d s" % (len(data_list), time.time() - time_start))
	print("ALL Done！")
