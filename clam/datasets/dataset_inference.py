from __future__ import print_function, division
import os
import numpy as np
import cv2
import openslide
import h5py
from .dataset_combine import Whole_Slide_Bag_COMBINE


class Whole_Slide_Bag_Infer(Whole_Slide_Bag_COMBINE):
    """用于推理的数据集"""

    def __init__(self, file_path, single_name, patch_level=0, patch_size=128,
                 slide_size=16, mode="lsil", transform=None):
        """
        Args:
            file_path (string): 数据对应目录
            single_name: 推理的文件名
            patch_level: wsi图像级别
            patch_size: 切片尺寸       
        """
        self.patch_level = patch_level
        self.patch_size = patch_size
        self.slide_size = slide_size
        self.file_path = file_path
        self.single_name = single_name
        self.transform = transform

        # 处理单独文件
        svs_file = os.path.join(file_path, "data", "{}.svs".format(single_name))
        patch_path = os.path.join(file_path, "patches_level{}".format(patch_level))
        patch_file = os.path.join(patch_path, single_name + ".h5")
        wsi_file = os.path.join(file_path, "data", svs_file)

        # 直接读取wsi文件，并根据h5中存储的坐标点进行分割
        wsi_data = openslide.open_slide(wsi_file)
        scale = wsi_data.level_downsamples[patch_level]
        slide_length = patch_size // slide_size

        with h5py.File(patch_file, "r") as f:
            # 图层为0的坐标
            patch_coords = np.array(f['coords'])
            patches_bag_list = []
            # 对每个patch坐标进行处理
            print("len:{},patch_level:{}".format(patch_coords.shape[0], patch_level))
            for i in range(patch_coords.shape[0]):
                # 尺寸缩放
                coord = patch_coords[i]
                coord = coord // scale
                coord = coord.astype(np.int16)
                # 每个patch区域内，再使用滑动窗进行目标区域框定
                for j in range(slide_length):
                    for k in range(slide_length):
                        coord_tar = np.array([coord[0] + j * patch_size, coord[1] + k * patch_size]).astype(np.int16)
                        # 预存pred字段，后续根据推理结果进行修改
                        patches_bag = {"pred": 0, "scale": scale, "name": single_name, "coord": coord,
                                       "coord_tar": coord_tar, "patch_level": patch_level}
                        patches_bag_list.append(patches_bag)

        self.patches_bag_list = patches_bag_list
        self.pathces_total_len = len(self.patches_bag_list)

    def get_wsi_obj(self):
        wsi_file = os.path.join(self.file_path, "data", self.single_name + ".svs")
        wsi = openslide.open_slide(wsi_file)
        return wsi

    def __len__(self):
        return self.pathces_total_len

    def __getitem__(self, idx):
        item = self.patches_bag_list[idx]
        name, coord, scale = item["name"], item["coord_tar"], item["scale"]

        # 读取wsi文件并生成图像数据
        wsi_file = os.path.join(self.file_path, "data", name + ".svs")
        wsi = openslide.open_slide(wsi_file)

        coord = (coord * scale).astype(np.int16)

        # 裁剪的起始坐标需是图层0的真是坐标
        img = wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if self.transform is not None:
            img_tar = self.transform(img)
        else:
            img_tar = img
        # 返回数据包括图像数据以及坐标
        return img_tar, coord, item, img
