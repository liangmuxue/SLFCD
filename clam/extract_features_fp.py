import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
import json
from argparse import Namespace
import torchvision.transforms as transforms

from clam.datasets.dataset_h5 import Dataset_Combine_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from clam.models.resnet_custom import resnet50_baseline
import argparse
from clam.utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
# import sys
# print("exit")
# sys.exit()
from custom.train_with_clamdata import CoolSystem, get_last_ck_file

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size=8, verbose=0, print_every=20, pretrained=True,
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""

	custom_transforms = transforms.Compose([
				transforms.ToTensor(),
				transforms.Resize(224),
				# normalize,
			]) 	
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained,custom_transforms=custom_transforms,
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path, len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			
			features,cls_emb = model(batch)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
# parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--level', type=int, default=1)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--types', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
args = parser.parse_args()

if __name__ == '__main__':

	print('initializing dataset')
	#cnn_path = 'custom/configs/config_lsil.json'
	cnn_path = 'custom/configs/config_{}_liang.json'.format(args.mode)
	with open(cnn_path, 'r') as f:
		att_args = json.load(f) 
	hparams = Namespace(**att_args) 
		
	data_path = args.data_dir
	types = args.types.split(",")
	bags_dataset = Dataset_Combine_Bags(data_path, types)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	for type in types:
		os.makedirs(os.path.join(args.feat_dir, 'pt_files', type), exist_ok=True)
		os.makedirs(os.path.join(args.feat_dir, 'h5_files', type), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint')
	checkpoint_path = os.path.join(hparams.work_dir, "checkpoints", hparams.model_name)
	file_name = get_last_ck_file(checkpoint_path)
	checkpoint_path_file = "{}/{}".format(checkpoint_path, file_name)
	model = CoolSystem.load_from_checkpoint(checkpoint_path_file).to(device)
	# Remove Fc layer
 # model = torch.nn.Sequential(*(list(model.model.children())[:-1]))
	
	# print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		
	model.eval()
	total = len(bags_dataset)
	
	for bag_candidate_idx in range(total):
		type, slide_id = bags_dataset[bag_candidate_idx]
		data_slide_dir = os.path.join(data_path, type, "data")
		data_h5_dir = os.path.join(data_path, type, "patches_level{}".format(args.level))
		slide_id = slide_id.split(args.slide_ext)[0]
		bag_name = slide_id + '.h5'
		bag_base, _ = os.path.splitext(bag_name)
		fea_file_path = os.path.join(args.feat_dir, 'pt_files', type, bag_base + '.pt')
		if os.path.exists(fea_file_path):
			print("file exists:{}".format(fea_file_path))
			continue
				
		h5_file_path = os.path.join(data_h5_dir, bag_name)
		slide_file_path = os.path.join(data_slide_dir, slide_id + args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))

		if not args.no_auto_skip and slide_id + '.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', type, bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		print("wsi", wsi)
		try:
			output_file_path = compute_w_loader(h5_file_path, output_path, wsi,
			model=model, batch_size=args.batch_size, verbose=1, print_every=20,
			custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
		except Exception as e:
			print("compute_w_loader fail:{}".format(bag_name))
			continue
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		torch.save(features, fea_file_path)

