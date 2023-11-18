from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

import openslide
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py

from random import randrange

def eval_transforms(pretrained=False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	trnsfrms_val = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean = mean, std = std)
					]
				)

	return trnsfrms_val


class Whole_Slide_Bag_COMBINE(Dataset):
	"""Custom slide dataset,use multiple wsi file,in which has multiple patches"""
	
	def __init__(self,
		file_path,
		wsi_path,
		mask_path,
		split_data=None,
		custom_downsample=1,
		target_patch_size=-1,
		patch_level=0,
		patch_size=256
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
		mask_data = {}
		pathces_total_len = 0
		# loop all patch files,and combine the coords data
		for svs_file in split_data:
			single_name = svs_file.split(".")[0]
			patch_file = os.path.join(file_path,"patches",single_name + ".h5")	
			wsi_file = os.path.join(file_path,"data",svs_file)	
			npy_file = single_name +  ".npy"
			npy_file = os.path.join(mask_path,npy_file)	
			mask_data[single_name] = np.load(npy_file)
			wsi_data[single_name] = openslide.open_slide(wsi_file)
			scale = wsi_data[single_name].level_downsamples[patch_level]
			with h5py.File(patch_file, "r") as f:
				self.patch_coords = np.array(f['coords'])
				patch_level = f['coords'].attrs['patch_level']
				patch_size = f['coords'].attrs['patch_size']
				
				# sum data length
				pathces_total_len += len(f['coords'])
				if target_patch_size > 0:
					target_patch_size = (target_patch_size, ) * 2
				elif custom_downsample > 1:
					target_patch_size = (patch_size // custom_downsample, ) * 2
				else:
					target_patch_size = None
				for coord in f['coords']:
					patches_bag = {"name":single_name}					
					patches_bag["coord"] = np.array(coord) /scale
					patches_bag["coord"] = patches_bag["coord"].astype(np.int16)
					patches_bag["patch_level"] = patch_level
					patches_bag_list.append(patches_bag)
		
				
		self.pathces_total_len = pathces_total_len
		self.patches_bag_list = patches_bag_list
		self.mask_data = mask_data
		
		self.roi_transforms = eval_transforms()
		self.target_patch_size = target_patch_size
		self.wsi_data = wsi_data
		
	def __len__(self):
		return self.pathces_total_len


	def __getitem__(self, idx):
		
		item = self.patches_bag_list[idx]
		coord = item['coord']
		name = item['name']
		wsi = self.wsi_data[name]
		# read image from wsi with coordination 
		img = wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		# get mask data and compute this region label
		masks = self.mask_data[name][coord[1]:coord[1]+self.patch_size,coord[0]:coord[0]+self.patch_size]
		mask_flag = masks[masks>0]
		mask_tumor_size = np.sum(mask_flag)
		img_size = self.patch_size * self.patch_size
		# if has a certain ratio mask flag in this region,then put this flag as region's label
		if mask_tumor_size/img_size > 0.3:
			label = np.argmax(np.bincount(mask_flag))
		else:
			label = 0
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = np.array(img)
		img = self.roi_transforms(img)
		return img,label





