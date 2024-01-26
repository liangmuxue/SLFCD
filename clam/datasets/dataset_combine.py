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
from utils.constance import get_tumor_label_cate


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
		# loop all patch files,and combine the coords data
		for svs_file in split_data:
			single_name = svs_file.split(".")[0]
			# lsil
			# if single_name == '62-CG23_14933_02':
			# 	continue
			# if single_name == '98-CG23_19585_02':
			# 	continue
			# if single_name == '86-CG23_18818_01':
			# 	continue
			file_names.append(single_name)
			patch_file = os.path.join(file_path, patch_path, single_name + ".h5")	
			wsi_file = os.path.join(file_path, "data", svs_file)	
			# lsil
			# if os.path.basename(wsi_file) == '49.svs':
			# 	continue
			# if os.path.basename(wsi_file) == '4-CG23 10032 01.svs':
			# 	continue
			# hsil
			if os.path.basename(wsi_file) == '100-CG23_15432_02.svs':
				continue
			npy_file = single_name + ".npy"
			npy_file = os.path.join(mask_path, npy_file)	
			wsi_data[single_name] = openslide.open_slide(wsi_file)
			scale = wsi_data[single_name].level_downsamples[patch_level]
			with h5py.File(patch_file, "r") as f:
				print(os.path.basename(patch_file))
				ignore_file = os.path.basename(patch_file)
				# lsil
				if ignore_file == '62-CG23_14933_02.h5':
					continue
				if ignore_file == '86-CG23_18818_01.h5':
					continue
				if ignore_file == '98-CG23_19585_02.h5':
					continue
				self.patch_coords = np.array(f['coords'])
				patch_level = f['coords'].attrs['patch_level']
				patch_size = f['coords'].attrs['patch_size']
				
				# sum data length
				pathces_normal_len += len(f['coords'])
				if target_patch_size > 0:
					target_patch_size = (target_patch_size,) * 2
				elif custom_downsample > 1:
					target_patch_size = (patch_size // custom_downsample,) * 2
					
				# Normal patch data
				for coord in f['coords']:
					patches_bag = {"name":single_name, "scale":scale, "type":"normal"}		
					patches_bag["coord"] = np.array(coord) / scale
					patches_bag["coord"] = patches_bag["coord"].astype(np.int16)
					patches_bag["patch_level"] = patch_level
					patches_bag["label"] = 0
					patches_bag_list.append(patches_bag)
					
			# Annotation patch data
			for label in get_tumor_label_cate():
				patch_img_path = None
				# Data augmentation
				if work_type == "train":
					if label == 4 or label == 5:
						patch_img_path = os.path.join(file_path, "tumor_patch_img", str(label), "output")
					elif label == 6:
						patch_img_path = os.path.join(file_path, "tumor_patch_img", str(label), "origin")
					elif label == 1 or label ==  2 or label ==  3:
						patch_img_path = os.path.join(file_path, "tumor_patch_img", str(label), "origin")
					
					
				else:
					patch_img_path = os.path.join(file_path, "tumor_patch_img", str(label), "origin")
				file_list = os.listdir(patch_img_path)
				for file in file_list:
					if not single_name in file:
						continue
					tumor_file_path = os.path.join(patch_img_path, file)
					patches_tumor_patch_file_list.append(tumor_file_path)
					pathces_tumor_len += 1
				
		self.patches_bag_list = patches_bag_list
		self.pathces_normal_len = pathces_normal_len					
		self.patches_tumor_patch_file_list = patches_tumor_patch_file_list
		self.pathces_tumor_len = pathces_tumor_len
				
		self.pathces_total_len = pathces_tumor_len + pathces_normal_len
		self.roi_transforms = eval_transforms()
		self.target_patch_size = target_patch_size
		
	def __len__(self):
		return self.pathces_total_len

	def __getitem__(self, idx):
		
		# Judge type by index value
		if idx >= self.pathces_normal_len:
			# print("mask_tumor_size is:{},coord:{}".format(mask_tumor_size,coord))
			file_path = self.patches_tumor_patch_file_list[idx - self.pathces_normal_len]
			t = file_path.split("/")
			try:
				label = int(t[-3])
				# label = 4
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
		
				
