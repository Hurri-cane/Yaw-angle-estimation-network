# author:Hurricane
# date:  2021/11/16
# E-mail:hurri_cane@qq.com
import torch.nn as nn
import torch


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


class Baseline1(nn.Module):
	def __init__(self):
		super(Baseline1, self).__init__()
		min_c = 64

		# Convolution by column
		self.conv_for_C = BasicConv(1, min_c, (6, 1))

		self.conv_for_R1 = BasicConv(min_c, min_c, (1, 1))
		self.conv_for_R2 = BasicConv(min_c, min_c * 2, (1, 2))
		self.conv_for_R3 = BasicConv(min_c, min_c * 1, (1, 3))
		self.conv_for_R3_1 = BasicConv(min_c, min_c * 2, (1, 3))
		self.conv_for_R4 = BasicConv(min_c, min_c * 2, (1, 4))
		self.conv_for_R4_1 = BasicConv(min_c, min_c * 4, (1, 4))

		self.upsample_for_R3 = Upsample(min_c * 2, min_c * 1, 2)
		self.upsample_for_R4 = Upsample(min_c * 2, min_c * 1, 2)

		self.conv_for_R1_R3 = BasicConv(min_c * 2, min_c * 4, (1, 4))
		self.conv_for_R3_R4 = BasicConv(min_c * 2, min_c * 4, (1, 2))

		self.conv_for_R2_1 = BasicConv(min_c * 2, min_c * 4, (1, 3))

		self.conv_for_R_all = BasicConv(min_c * 16, min_c * 16, (1, 1))

		self.conv_for_all = BasicConv(min_c * 16, min_c * 16, (1, 1))
		self.fconnect = nn.Sequential(
			# nn.Linear(min_c * 16, 2048),
			nn.LeakyReLU(0.1),
			nn.Dropout(0.5),
			nn.Linear(min_c * 16, 1),
		)

	def forward(self, h):
		C = self.conv_for_C(h)
		R1 = self.conv_for_R1(C)
		R2 = self.conv_for_R2(C)
		R3 = self.conv_for_R3(C)
		R4 = self.conv_for_R4(C)
		R3_1 = self.conv_for_R3_1(C)
		R4_1 = self.conv_for_R4_1(C)

		R3_Upsample = self.upsample_for_R3(R3_1)[:, :, 0, :].unsqueeze(2)
		R4_Upsample = self.upsample_for_R4(R4)[:, :, 0, :].unsqueeze(2)

		R1_R3 = torch.cat([R1, R3_Upsample], axis=1)
		R3_R4 = torch.cat([R3, R4_Upsample], axis=1)

		R1_R3 = self.conv_for_R1_R3(R1_R3)
		R3_R4 = self.conv_for_R3_R4(R3_R4)
		R2 = self.conv_for_R2_1(R2)

		R_all = torch.cat([R1_R3, R3_R4, R2, R4_1], axis=1)
		R_all = self.conv_for_R_all(R_all)
		out = R_all
		out = self.conv_for_all(out)[:, :, 0, 0]
		out = self.fconnect(out)

		return out


class Baseline2(nn.Module):
	def __init__(self):
		super(Baseline2, self).__init__()
		min_c = 64
		# Convolution by row
		self.conv_for_R = BasicConv(1, min_c, (1, 4))

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
			nn.Linear(min_c * 16, 1),
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


class Baseline3(nn.Module):
	def __init__(self):
		super(Baseline3, self).__init__()
		min_r = 64
		min_c = 32
		# Convolution by row
		self.conv_for_R = BasicConv(1, min_r, (1, 4))

		self.conv_for_C1 = BasicConv(min_r, min_r, (1, 1))
		self.conv_for_C2 = BasicConv(min_r, min_r * 2, (2, 1))
		self.conv_for_C3 = BasicConv(min_r, min_r, (3, 1))
		self.conv_for_C4 = BasicConv(min_r, min_r * 2, (4, 1))
		self.conv_for_C5 = BasicConv(min_r, min_r * 2, (5, 1))
		self.conv_for_C6 = BasicConv(min_r, min_r * 4, (6, 1))

		self.upsample_for_C4 = Upsample(min_r * 2, min_r * 1, 2)
		self.upsample_for_C5 = Upsample(min_r * 2, min_r * 1, 2)

		self.conv_for_C1_C4 = BasicConv(min_r * 2, min_r * 4, (6, 1))
		self.conv_for_C3_C5 = BasicConv(min_r * 2, min_r * 4, (4, 1))
		self.conv_for_C2_1 = BasicConv(min_r * 2, min_r * 4, (5, 1))

		self.conv_for_C_all = BasicConv(min_r * 16, min_r * 16, (1, 1))

		# Convolution by column
		self.conv_for_C = BasicConv(1, min_c, (6, 1))

		self.conv_for_R1 = BasicConv(min_c, min_c, (1, 1))
		self.conv_for_R2 = BasicConv(min_c, min_c * 2, (1, 2))
		self.conv_for_R3 = BasicConv(min_c, min_c * 1, (1, 3))
		self.conv_for_R3_1 = BasicConv(min_c, min_c * 2, (1, 3))
		self.conv_for_R4 = BasicConv(min_c, min_c * 2, (1, 4))
		self.conv_for_R4_1 = BasicConv(min_c, min_c * 4, (1, 4))

		self.upsample_for_R3 = Upsample(min_c * 2, min_c * 1, 2)
		self.upsample_for_R4 = Upsample(min_c * 2, min_c * 1, 2)

		self.conv_for_R1_R3 = BasicConv(min_c * 2, min_c * 4, (1, 4))
		self.conv_for_R3_R4 = BasicConv(min_c * 2, min_c * 4, (1, 2))

		self.conv_for_R2_1 = BasicConv(min_c * 2, min_c * 4, (1, 3))

		self.conv_for_R_all = BasicConv(min_c * 16, min_c * 16, (1, 1))

		self.conv_for_all = BasicConv(min_r * 16 + min_c * 16, min_c * 16, (1, 1))
		self.fconnect = nn.Sequential(
			# nn.Linear(min_c * 16, 2048),
			nn.LeakyReLU(0.1),
			nn.Dropout(0.5),
			nn.Linear(min_c * 16, 1),
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

		C = self.conv_for_C(h)
		R1 = self.conv_for_R1(C)
		R2 = self.conv_for_R2(C)
		R3 = self.conv_for_R3(C)
		R4 = self.conv_for_R4(C)
		R3_1 = self.conv_for_R3_1(C)
		R4_1 = self.conv_for_R4_1(C)

		R3_Upsample = self.upsample_for_R3(R3_1)[:, :, 0, :].unsqueeze(2)
		R4_Upsample = self.upsample_for_R4(R4)[:, :, 0, :].unsqueeze(2)

		R1_R3 = torch.cat([R1, R3_Upsample], axis=1)
		R3_R4 = torch.cat([R3, R4_Upsample], axis=1)

		R1_R3 = self.conv_for_R1_R3(R1_R3)
		R3_R4 = self.conv_for_R3_R4(R3_R4)
		R2 = self.conv_for_R2_1(R2)

		R_all = torch.cat([R1_R3, R3_R4, R2, R4_1], axis=1)
		R_all = self.conv_for_R_all(R_all)

		out = torch.cat([R_all, C_all], axis=1)
		out = self.conv_for_all(out)[:, :, 0, 0]
		out = self.fconnect(out)

		return out


class Baseline4(nn.Module):
	def __init__(self):
		super(Baseline4, self).__init__()
		min_c = 64
		# Convolution by row
		self.conv_for_R = BasicConv(1, min_c, (1, 4))

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
			nn.Linear(min_c * 16, 1),
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


class Baseline5(nn.Module):
	def __init__(self):
		super(Baseline5, self).__init__()
		min_c = 64
		# Convolution by row
		self.conv_for_R = BasicConv(1, min_c, (1, 4))

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


class Baseline6(nn.Module):
	def __init__(self):
		super(Baseline6, self).__init__()
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


class Baseline7(nn.Module):
	def __init__(self):
		super(Baseline7, self).__init__()
		min_c = 32
		# Convolution by row
		self.conv_for_R = BasicConv(1, min_c, (1, 4))

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

		# Convolution by column
		self.conv_for_C = BasicConv(1, min_c, (6, 1))

		self.conv_for_R1 = BasicConv(min_c, min_c, (1, 1))
		self.conv_for_R2 = BasicConv(min_c, min_c * 2, (1, 2))
		self.conv_for_R3 = BasicConv(min_c, min_c * 1, (1, 3))
		self.conv_for_R3_1 = BasicConv(min_c, min_c * 2, (1, 3))
		self.conv_for_R4 = BasicConv(min_c, min_c * 2, (1, 4))
		self.conv_for_R4_1 = BasicConv(min_c, min_c * 4, (1, 4))

		self.upsample_for_R3 = Upsample(min_c * 2, min_c * 1, 2)
		self.upsample_for_R4 = Upsample(min_c * 2, min_c * 1, 2)

		self.conv_for_R1_R3 = BasicConv(min_c * 2, min_c * 4, (1, 4))
		self.conv_for_R3_R4 = BasicConv(min_c * 2, min_c * 4, (1, 2))

		self.conv_for_R2_1 = BasicConv(min_c * 2, min_c * 4, (1, 3))

		self.conv_for_R_all = BasicConv(min_c * 16, min_c * 16, (1, 1))

		self.conv_for_all = BasicConv(min_c * 32, min_c * 16, (1, 1))
		self.fconnect = nn.Sequential(
			# nn.Linear(min_c * 16, 2048),
			nn.LeakyReLU(0.1),
			nn.Dropout(0.5),
			nn.Linear(min_c * 16, 1),
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

		C = self.conv_for_C(h)
		R1 = self.conv_for_R1(C)
		R2 = self.conv_for_R2(C)
		R3 = self.conv_for_R3(C)
		R4 = self.conv_for_R4(C)
		R3_1 = self.conv_for_R3_1(C)
		R4_1 = self.conv_for_R4_1(C)

		R3_Upsample = self.upsample_for_R3(R3_1)[:, :, 0, :].unsqueeze(2)
		R4_Upsample = self.upsample_for_R4(R4)[:, :, 0, :].unsqueeze(2)

		R1_R3 = torch.cat([R1, R3_Upsample], axis=1)
		R3_R4 = torch.cat([R3, R4_Upsample], axis=1)

		R1_R3 = self.conv_for_R1_R3(R1_R3)
		R3_R4 = self.conv_for_R3_R4(R3_R4)
		R2 = self.conv_for_R2_1(R2)

		R_all = torch.cat([R1_R3, R3_R4, R2, R4_1], axis=1)
		R_all = self.conv_for_R_all(R_all)

		out = torch.cat([R_all, C_all], axis=1)
		out = self.conv_for_all(out)[:, :, 0, 0]
		out = self.fconnect(out)

		return out


class Baseline8(nn.Module):
	def __init__(self):
		super(Baseline8, self).__init__()
		min_c = 32
		# Convolution by row
		self.conv_for_R = BasicConv(1, min_c, (1, 4))

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

		# Convolution by column
		self.conv_for_C = BasicConv(1, min_c, (6, 1))

		self.conv_for_R1 = BasicConv(min_c, min_c, (1, 1))
		self.conv_for_R2 = BasicConv(min_c, min_c * 2, (1, 2))
		self.conv_for_R3 = BasicConv(min_c, min_c * 1, (1, 3))
		self.conv_for_R3_1 = BasicConv(min_c, min_c * 2, (1, 3))
		self.conv_for_R4 = BasicConv(min_c, min_c * 2, (1, 4))
		self.conv_for_R4_1 = BasicConv(min_c, min_c * 4, (1, 4))

		self.upsample_for_R3 = Upsample(min_c * 2, min_c * 1, 2)
		self.upsample_for_R4 = Upsample(min_c * 2, min_c * 1, 2)

		self.conv_for_R1_R3 = BasicConv(min_c * 2, min_c * 4, (1, 4))
		self.conv_for_R3_R4 = BasicConv(min_c * 2, min_c * 4, (1, 2))

		self.conv_for_R2_1 = BasicConv(min_c * 2, min_c * 4, (1, 3))

		self.conv_for_R_all = BasicConv(min_c * 16, min_c * 16, (1, 1))

		self.conv_for_all = BasicConv(min_c * 32, min_c * 16, (1, 1))
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

		C = self.conv_for_C(h)
		R1 = self.conv_for_R1(C)
		R2 = self.conv_for_R2(C)
		R3 = self.conv_for_R3(C)
		R4 = self.conv_for_R4(C)
		R3_1 = self.conv_for_R3_1(C)
		R4_1 = self.conv_for_R4_1(C)

		R3_Upsample = self.upsample_for_R3(R3_1)[:, :, 0, :].unsqueeze(2)
		R4_Upsample = self.upsample_for_R4(R4)[:, :, 0, :].unsqueeze(2)

		R1_R3 = torch.cat([R1, R3_Upsample], axis=1)
		R3_R4 = torch.cat([R3, R4_Upsample], axis=1)

		R1_R3 = self.conv_for_R1_R3(R1_R3)
		R3_R4 = self.conv_for_R3_R4(R3_R4)
		R2 = self.conv_for_R2_1(R2)

		R_all = torch.cat([R1_R3, R3_R4, R2, R4_1], axis=1)
		R_all = self.conv_for_R_all(R_all)

		out = torch.cat([R_all, C_all], axis=1)
		out = self.conv_for_all(out)[:, :, 0, 0]
		out = self.fconnect(out)

		return out


# train_vH_C_one_hot_add_single_new_way_V4
class Baseline9(nn.Module):
	def __init__(self):
		super(Baseline9, self).__init__()
		min_c = 32
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

		self.conv_for_C_all = BasicConv(min_c * 16, min_c * 8, (1, 1))

		# Convolution by column
		self.conv_for_C = BasicConv(1, min_c, (6, 1))

		self.conv_for_R1 = BasicConv(min_c, min_c, (1, 1))
		self.conv_for_R2 = BasicConv(min_c, min_c * 2, (1, 2))
		self.conv_for_R3 = BasicConv(min_c, min_c, (1, 3))
		self.conv_for_R4 = BasicConv(min_c, min_c * 2, (1, 4))
		self.conv_for_R5 = BasicConv(min_c, min_c * 2, (1, 5))
		self.conv_for_R5_1 = BasicConv(min_c, min_c, (1, 5))
		self.conv_for_R6 = BasicConv(min_c, min_c * 2, (1, 6))
		self.conv_for_R7 = BasicConv(min_c, min_c * 2, (1, 7))
		self.conv_for_R8 = BasicConv(min_c, min_c * 4, (1, 8))

		self.upsample_for_R5 = Upsample(min_c * 2, min_c * 1, 2)
		self.upsample_for_R6 = Upsample(min_c * 2, min_c * 1, 2)
		self.upsample_for_R7 = Upsample(min_c * 2, min_c * 1, 2)

		self.conv_for_R1_R5 = BasicConv(min_c * 2, min_c * 4, (1, 8))
		self.conv_for_R3_R6 = BasicConv(min_c * 2, min_c * 4, (1, 6))
		self.conv_for_R5_1_R7 = BasicConv(min_c * 2, min_c * 4, (1, 4))
		self.conv_for_R2_1 = BasicConv(min_c * 2, min_c * 4, (1, 7))
		self.conv_for_R4_1 = BasicConv(min_c * 2, min_c * 4, (1, 5))

		self.conv_for_R_all = BasicConv(min_c * 24, min_c * 8, (1, 1))

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

		C = self.conv_for_C(h)
		R1 = self.conv_for_R1(C)
		R2 = self.conv_for_R2(C)
		R3 = self.conv_for_R3(C)
		R4 = self.conv_for_R4(C)
		R5 = self.conv_for_R5(C)
		R5_1 = self.conv_for_R5_1(C)
		R6 = self.conv_for_R6(C)
		R7 = self.conv_for_R7(C)
		R8 = self.conv_for_R8(C)

		R5_Upsample = self.upsample_for_R5(R5)[:, :, 0, :].unsqueeze(2)
		R6_Upsample = self.upsample_for_R6(R6)[:, :, 0, :].unsqueeze(2)
		R7_Upsample = self.upsample_for_R7(R7)[:, :, 0, :].unsqueeze(2)

		R1_R5 = torch.cat([R1, R5_Upsample], axis=1)
		R3_R6 = torch.cat([R3, R6_Upsample], axis=1)
		R5_R7 = torch.cat([R5_1, R7_Upsample], axis=1)

		R1_R5 = self.conv_for_R1_R5(R1_R5)
		R3_R6 = self.conv_for_R3_R6(R3_R6)
		R5_R7 = self.conv_for_R5_1_R7(R5_R7)
		R2 = self.conv_for_R2_1(R2)
		R4 = self.conv_for_R4_1(R4)

		R_all = torch.cat([R1_R5, R3_R6, R5_R7, R2, R4, R8], axis=1)
		R_all = self.conv_for_R_all(R_all)
		out = torch.cat([R_all, C_all], axis=1)
		out = self.conv_for_all(out)[:, :, 0, 0]
		out = self.fconnect(out)

		return out


class Baseline10(nn.Module):
	def __init__(self):
		super(Baseline10, self).__init__()
		min_c = 32
		# Convolution by row
		self.conv_for_R = BasicConv(1, min_c, (1, 8))

		self.conv_for_C1 = BasicConv(min_c, min_c * 1, (1, 1))
		self.conv_for_C2 = BasicConv(min_c, min_c * 2, (2, 1))
		self.conv_for_C3 = BasicConv(min_c, min_c * 1, (3, 1))
		self.conv_for_C4 = BasicConv(min_c, min_c * 2, (4, 1))
		self.conv_for_C5 = BasicConv(min_c, min_c * 2, (5, 1))
		self.conv_for_C6 = BasicConv(min_c, min_c * 4, (6, 1))
		self.conv_for_C5_1 = BasicConv(min_c, min_c * 1, (5, 1))
		self.conv_for_C6_1 = BasicConv(min_c, min_c * 2, (6, 1))

		self.upsample_for_C4 = Upsample(min_c * 2, min_c * 1, 2)
		self.upsample_for_C5 = Upsample(min_c * 2, min_c * 1, 2)
		self.upsample_for_C6 = Upsample(min_c * 2, min_c * 1, 2)

		self.conv_for_C1_C4 = BasicConv(min_c * 2, min_c * 4, (6, 1))
		self.conv_for_C3_C5 = BasicConv(min_c * 2, min_c * 4, (4, 1))
		self.conv_for_C5_C6 = BasicConv(min_c * 2, min_c * 4, (2, 1))
		self.conv_for_C2_1 = BasicConv(min_c * 2, min_c * 4, (5, 1))
		self.conv_for_C6_2 = BasicConv(min_c * 4, min_c * 4, (1, 1))

		self.conv_for_C_all = BasicConv(min_c * 20, min_c * 16, (1, 1))

		# Convolution by column
		self.conv_for_C = BasicConv(1, min_c, (6, 1))

		self.conv_for_R1 = BasicConv(min_c, min_c * 1, (1, 1))
		self.conv_for_R2 = BasicConv(min_c, min_c * 2, (1, 2))
		self.conv_for_R3 = BasicConv(min_c, min_c * 1, (1, 3))
		self.conv_for_R4 = BasicConv(min_c, min_c * 2, (1, 4))
		self.conv_for_R5 = BasicConv(min_c, min_c * 2, (1, 5))
		self.conv_for_R6 = BasicConv(min_c, min_c * 2, (1, 6))
		self.conv_for_R7 = BasicConv(min_c, min_c * 2, (1, 7))
		self.conv_for_R8 = BasicConv(min_c, min_c * 4, (1, 8))
		self.conv_for_R5_1 = BasicConv(min_c, min_c * 1, (1, 5))
		self.conv_for_R7_1 = BasicConv(min_c, min_c * 1, (1, 7))
		self.conv_for_R8_1 = BasicConv(min_c, min_c * 2, (1, 8))

		self.upsample_for_R5 = Upsample(min_c * 2, min_c * 1, 2)
		self.upsample_for_R6 = Upsample(min_c * 2, min_c * 1, 2)
		self.upsample_for_R7 = Upsample(min_c * 2, min_c * 1, 2)
		self.upsample_for_R8 = Upsample(min_c * 2, min_c * 1, 2)

		self.conv_for_R1_R5 = BasicConv(min_c * 2, min_c * 4, (1, 8))
		self.conv_for_R3_R6 = BasicConv(min_c * 2, min_c * 4, (1, 6))
		self.conv_for_R5_R7 = BasicConv(min_c * 2, min_c * 4, (1, 4))
		self.conv_for_R7_R8 = BasicConv(min_c * 2, min_c * 4, (1, 2))
		self.conv_for_R2_1 = BasicConv(min_c * 2, min_c * 4, (1, 7))
		self.conv_for_R4_1 = BasicConv(min_c * 2, min_c * 4, (1, 5))
		self.conv_for_R8_2 = BasicConv(min_c * 4, min_c * 4, (1, 1))

		self.conv_for_R_all = BasicConv(min_c * 28, min_c * 16, (1, 1))

		self.conv_for_all = BasicConv(min_c * 32, min_c * 16, (1, 1))
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
		C5_1 = self.conv_for_C5_1(R)
		C6_1 = self.conv_for_C6_1(R)

		C4_Upsample = self.upsample_for_C4(C4)[:, :, :, 0].unsqueeze(3)
		C5_Upsample = self.upsample_for_C5(C5)[:, :, :, 0].unsqueeze(3)
		C6_Upsample = self.upsample_for_C6(C6_1)[:, :, :, 0].unsqueeze(3)

		C1_C4 = torch.cat([C1, C4_Upsample], axis=1)
		C3_C5 = torch.cat([C3, C5_Upsample], axis=1)
		C5_C6 = torch.cat([C5_1, C6_Upsample], axis=1)

		C1_C4 = self.conv_for_C1_C4(C1_C4)
		C3_C5 = self.conv_for_C3_C5(C3_C5)
		C5_C6 = self.conv_for_C5_C6(C5_C6)
		C2 = self.conv_for_C2_1(C2)
		C6 = self.conv_for_C6_2(C6)

		C_all = torch.cat([C1_C4, C3_C5, C5_C6, C2, C6], axis=1)
		C_all = self.conv_for_C_all(C_all)

		C = self.conv_for_C(h)
		R1 = self.conv_for_R1(C)
		R2 = self.conv_for_R2(C)
		R3 = self.conv_for_R3(C)
		R4 = self.conv_for_R4(C)
		R5 = self.conv_for_R5(C)
		R6 = self.conv_for_R6(C)
		R7 = self.conv_for_R7(C)
		R8 = self.conv_for_R8(C)
		R5_1 = self.conv_for_R5_1(C)
		R7_1 = self.conv_for_R7_1(C)
		R8_1 = self.conv_for_R8_1(C)

		R5_Upsample = self.upsample_for_R5(R5)[:, :, 0, :].unsqueeze(2)
		R6_Upsample = self.upsample_for_R6(R6)[:, :, 0, :].unsqueeze(2)
		R7_Upsample = self.upsample_for_R7(R7)[:, :, 0, :].unsqueeze(2)
		R8_Upsample = self.upsample_for_R8(R8_1)[:, :, 0, :].unsqueeze(2)

		R1_R5 = torch.cat([R1, R5_Upsample], axis=1)
		R3_R6 = torch.cat([R3, R6_Upsample], axis=1)
		R5_R7 = torch.cat([R5_1, R7_Upsample], axis=1)
		R7_R8 = torch.cat([R7_1, R8_Upsample], axis=1)

		R1_R5 = self.conv_for_R1_R5(R1_R5)
		R3_R6 = self.conv_for_R3_R6(R3_R6)
		R5_R7 = self.conv_for_R5_R7(R5_R7)
		R7_R8 = self.conv_for_R7_R8(R7_R8)
		R2 = self.conv_for_R2_1(R2)
		R4 = self.conv_for_R4_1(R4)
		R8 = self.conv_for_R8_2(R8)

		R_all = torch.cat([R1_R5, R3_R6, R5_R7, R7_R8, R2, R4, R8], axis=1)
		R_all = self.conv_for_R_all(R_all)
		out = torch.cat([R_all, C_all], axis=1)
		out = self.conv_for_all(out)[:, :, 0, 0]
		out = self.fconnect(out)

		return out
