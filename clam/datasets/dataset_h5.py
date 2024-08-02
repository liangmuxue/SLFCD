from __future__ import print_function, division
import os

import cv2
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import h5py


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


class Whole_Slide_Bag(Dataset):
    def __init__(self, file_path, pretrained=False, custom_transforms=None, target_patch_size=-1):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
        """

        self.pretrained = pretrained
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None

        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['imgs']
        for name, value in dset.attrs.items():
            print(name, value)

        print('pretrained:', self.pretrained)
        print('transformations:', self.roi_transforms)
        if self.target_patch_size is not None:
            print('target_size: ', self.target_patch_size)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
            coord = hdf5_file['coords'][idx]

        img = Image.fromarray(img)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord


class Whole_Slide_Bag_FP(Dataset):
    def __init__(self, file_path, wsi, pretrained=False, custom_transforms=None, custom_downsample=1,
                 target_patch_size=-1):
        """
        处理全景图像
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.pretrained = pretrained
        self.wsi = wsi
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path
        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size,) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample,) * 2
            else:
                self.target_patch_size = None
        # self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord


class Whole_Slide_Bag_FP_all(Dataset):
    def __init__(self, file_path, wsi, patch_level, patch_size, slide_size, transforms=None):
        self.file_path = file_path
        self.wsi = wsi
        self.patch_level = patch_level
        self.patch_size = patch_size
        self.slide_size = slide_size
        self.transforms = transforms

        slide_length = self.patch_size // self.slide_size

        image = np.array(wsi.read_region((0, 0), self.patch_level, wsi.level_dimensions[self.patch_level]))
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        scale, shape = wsi.level_downsamples[self.patch_level], wsi.level_dimensions[self.patch_level]

        self.patches_bag_list = []
        with h5py.File(self.file_path, "r") as f:
            coords = np.array(f['coords'])
            for coord in coords:
                coord = np.array(coord / scale).astype(np.int32)
                for j in range(slide_length):
                    for k in range(slide_length):
                        coord_tar = np.array([coord[0] + j * self.patch_size, coord[1] + k * self.patch_size]).astype(
                            np.int16)
                        coord_tar = [coord_tar[0], coord_tar[1], coord_tar[0] + self.patch_size,
                                     coord_tar[1] + self.patch_size]
                        if coord_tar[2] < shape[0] and coord_tar[3] < shape[1]:
                            self.patches_bag_list.append(coord_tar)

    def __len__(self):
        return self.patches_bag_list

    def __getitem__(self, idx):
        coord_tar = self.patches_bag_list[idx]
        img = self.image[coord_tar[1]:coord_tar[3], coord_tar[0]:coord_tar[2], :]

        img_tar = cv2.resize(img, (224, 224))
        if self.transforms:
            img_tar = self.transforms(img_tar)
        return img_tar, torch.tensor(coord_tar)


class Dataset_All_Bags(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]


class Dataset_Combine_Bags(Dataset):
    """
        Custom dataset,combine multiple type data
    """
    def __init__(self, parent_path, types):
        self.data_df = None

        for type in types:
            # 获取每个类别中的数据
            item_path = os.path.join(parent_path, type)
            csv_path = os.path.join(item_path, "process_list_autogen.csv")
            df = pd.read_csv(csv_path)

            # Add type column
            df.insert(df.shape[1], 'type', type)
            if self.data_df is None:
                self.data_df = df
            else:
                self.data_df = pd.concat([self.data_df, df])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        return self.data_df['type'].values[idx], self.data_df['slide_id'].values[idx]
