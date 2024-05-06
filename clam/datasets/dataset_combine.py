from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
import cv2
import openslide
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py
from random import randrange
from utils.constance import get_label_cate_num,get_tumor_label_cate


def eval_transforms(pretrained=False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

	else:
		mean = (0.5, 0.5, 0.5)
		std = (0.5, 0.5, 0.5)

	trnsfrms_val = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean=mean, std=std)
					]
				)

	return trnsfrms_val


class Whole_Slide_Bag_COMBINE(Dataset):
	"""Custom slide dataset,use multiple wsi file,in which has multiple patches"""
	
	def __init__(self,
		file_path,
		wsi_path,
		mask_path,
		patch_path=None,
		split_data=None,
		custom_downsample=1,
		target_patch_size=-1,
		patch_level=0,
		patch_size=256,
		mode="lsil",
		work_type="train",
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			wsi_path: Path to the .wsi file containing wsi data.
			mask_path: Path to the mask file containing tumor annotation mask data.
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.patch_level = patch_level
		self.patch_size = patch_size
		self.file_path = file_path
		self.wsi_path = wsi_path
		self.mask_path = mask_path

		wsi_data = {}
		patches_bag_list = []
		patches_tumor_patch_file_list = []
		pathces_normal_len = 0
		pathces_tumor_len = 0
		file_names = []
		
		# 处理非标注数据，遍历所有h5数据，存储相关坐标
		for svs_file in split_data:
			single_name = svs_file.split(".")[0]
			file_names.append(single_name)
			patch_file = os.path.join(file_path, patch_path, single_name + ".h5")	
			wsi_file = os.path.join(file_path, "data", svs_file)	
			wsi_data[single_name] = openslide.open_slide(wsi_file)
			scale = wsi_data[single_name].level_downsamples[patch_level]
			with h5py.File(patch_file, "r") as f:
				ignore_file = os.path.basename(patch_file)
				try:
					self.patch_coords = np.array(f['coords'])
				except Exception as e:
					print("file load fail:{}".format(patch_file))
					continue
				patch_level = f['coords'].attrs['patch_level']
				patch_size = f['coords'].attrs['patch_size']
				
				# sum data length
				pathces_normal_len += len(f['coords'])
				if target_patch_size > 0:
					# 如果目标分片尺寸大于0，需要对应处理，一般用不到
					target_patch_size = (target_patch_size,) * 2
				elif custom_downsample > 1:
					target_patch_size = (patch_size // custom_downsample,) * 2

				# 对每个patch坐标进行处理，包括设置缩放等
				for coord in f['coords']:
# # 通过标注掩码信息，判断所属区域类型 
# coord_x = int(coord[0]/scale)
# coord_y = int(coord[1]/scale)
# mask_data_item = mask_data[coord_y:coord_y+patch_size,coord_x:coord_x+patch_size]	
# # 超出标注数量阈值，则视为标注类别
# if np.sum(mask_data_item>0)>100:
# 	patches_bag["type"] = "tumor"		
# 	patches_bag["label"] = 1						
# else:
# 	patches_bag["type"] = "normal"	
# 	patches_bag["label"] = 0
																
					patches_bag = {"name":single_name, "scale":scale, "type":"normal"}		
					patches_bag["coord"] = np.array(coord) / scale
					patches_bag["coord"] = patches_bag["coord"].astype(np.int16)
					patches_bag["patch_level"] = patch_level
					patches_bag["label"] = 0
					patches_bag_list.append(patches_bag)
					
		# 处理标注数据
		for label in get_tumor_label_cate(mode=mode):
			patch_img_path = None
			# Data augmentation
			if work_type == "train":
				if label == 4 or label == 5:
					patch_img_path = os.path.join(file_path, "tumor_patch_img", str(label), "origin")
				elif label == 6:
					patch_img_path = os.path.join(file_path, "tumor_patch_img", str(label), "origin")
				elif label == 1 or label ==  2 or label ==  3:
					patch_img_path = os.path.join(file_path, "tumor_patch_img", str(label), "origin")
			else:
				patch_img_path = os.path.join(file_path, "tumor_patch_img", str(label), "origin")
			file_list = os.listdir(patch_img_path)
			for file in file_list:
				img_file_name = file.split(".")[0]
				match_svs_file_name = img_file_name.rsplit("_",1)[0]
				# 检查当前标注的图片文件是否属于此分割数据集合中的
				if not match_svs_file_name in file_names:
					continue
				tumor_file_path = os.path.join(patch_img_path, file)
				patches_tumor_patch_file_list.append(tumor_file_path)
				pathces_tumor_len += 1
		
		if work_type=="train":
			# 削减非标注样本数量，以达到数据规模平衡
			target_pathces_normal_len = pathces_tumor_len * 5
			target_pathces_normal_idx = np.random.randint(0,pathces_normal_len-1,(target_pathces_normal_len,))
			self.patches_bag_list = np.array(patches_bag_list)[target_pathces_normal_idx].tolist()
			self.pathces_normal_len = target_pathces_normal_len	
		else:
			target_pathces_normal_len = pathces_tumor_len * 10
			target_pathces_normal_idx = np.random.randint(0,pathces_normal_len-1,(target_pathces_normal_len,))
			self.patches_bag_list = np.array(patches_bag_list)[target_pathces_normal_idx].tolist()
			self.pathces_normal_len = target_pathces_normal_len	
		
		self.pathces_normal_len = target_pathces_normal_len					
		self.patches_tumor_patch_file_list = patches_tumor_patch_file_list
		self.pathces_tumor_len = pathces_tumor_len
				
		self.pathces_total_len = pathces_tumor_len + target_pathces_normal_len
		self.roi_transforms = eval_transforms()
		self.target_patch_size = target_patch_size
		
	def __len__(self):
		return self.pathces_total_len

	def __getitem__(self, idx):
		
		# Judge type by index value
		if idx >= self.pathces_normal_len:
			# print("mask_tumor_size is:{},coord:{}".format(mask_tumor_size,coord))
			file_path = self.patches_tumor_patch_file_list[idx - self.pathces_normal_len]
			# 通过文件名中的标志，取得对应标签代码
			t = file_path.split("/")
			try:
				label = int(t[-3])
				# 标签代码转为标签序号
				label = get_label_cate_num(label)
			except Exception as e:
				print("sp err:{}".format(t))
			img_ori = cv2.imread(file_path)
			item = {}
		else:
			item = self.patches_bag_list[idx]
			name = item['name']
			scale = item['scale']
			coord = item['coord']
			wsi_file = os.path.join(self.file_path, "data", name + ".svs")	
			wsi = openslide.open_slide(wsi_file)			
			# read image from wsi with coordination 
			coord_ori = (coord * scale).astype(np.int16)			
			img_ori = wsi.read_region(coord_ori, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
			img_ori = cv2.cvtColor(np.array(img_ori), cv2.COLOR_RGB2BGR)	
			label = 0
		if self.target_patch_size > 0:
			img_ori = img_ori.resize(self.target_patch_size)
		img = self.roi_transforms(img_ori)
		return img, label, img_ori, item
		
				
