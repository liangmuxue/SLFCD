from tqdm import tqdm
import os
import numpy as np
import cv2
import openslide
import h5py
from .dataset_combine import Whole_Slide_Bag_COMBINE


class Whole_Slide_Bag_Infer(Whole_Slide_Bag_COMBINE):
    """用于推理的数据集"""

    def __init__(self, file_path, single_name, patch_level=0, patch_size=128,
                 slide_size=16, mode="lsil", transform=None, result_path=None):
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

        orig_img = wsi_data.read_region((0, 0), patch_level, wsi_data.level_dimensions[patch_level]).convert("RGB")
        orig_img = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)

        with h5py.File(patch_file, "r") as f:
            # 图层为0的坐标
            patch_coords = np.array(f['coords'])
            patches_bag_list = []
            # 对每个patch坐标进行处理
            # print("len:{},patch_level:{}".format(patch_coords.shape[0], patch_level))
            for i in range(patch_coords.shape[0]):
                # 尺寸缩放
                coord = patch_coords[i]
                cv2.putText(orig_img, "1",
                            (int((coord[0] + patch_size / 2) / scale),
                             int((coord[1] + patch_size / 2) / scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 0, 0), thickness=10)

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

            if not os.path.exists(f"{result_path}/{single_name}"):
                os.makedirs(f"{result_path}/{single_name}")
            cv2.imwrite(f"{result_path}/{single_name}/{single_name}_img.jpg", orig_img)

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


class Whole_Slide_Bag_Infer_all(Whole_Slide_Bag_COMBINE):
    """
    用于推理的数据集
    """
    def __init__(self, file_path, single_name, patch_level=0, patch_size=128,
                 slide_size=16, transform=None, result_path=None):
        self.patch_level = patch_level
        self.patch_size = patch_size
        self.slide_size = slide_size
        self.file_path = file_path
        self.single_name = single_name
        self.transform = transform

        # if not os.path.exists(f"{result_path}/{single_name}/infer"):
        #     os.makedirs(f"{result_path}/{single_name}/infer")
            
        # 处理单独文件
        patch_file = os.path.join(file_path, "patches_level{}".format(patch_level), single_name + ".h5")
        wsi_file = os.path.join(file_path, "data", "{}.svs".format(single_name))

        # 直接读取wsi文件，并根据h5中存储的坐标点进行分割
        wsi_data = openslide.open_slide(wsi_file)
        scale = wsi_data.level_downsamples[patch_level]
        slide_length = patch_size // slide_size

        image = np.array(wsi_data.read_region((0, 0), self.patch_level, wsi_data.level_dimensions[self.patch_level]))
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        orig_img = self.image.copy()
        self.patches_bag_list = []
        with h5py.File(patch_file, "r") as f:
            patch_coords = np.array(f['coords'])
            for coord in tqdm(patch_coords, total=patch_coords.shape[0], desc="process svs"):
                coord = np.array(coord / scale).astype(np.int32)

                for j in range(slide_length):
                    for k in range(slide_length):
                        coord_tar = np.array([coord[0] + j * patch_size, coord[1] + k * patch_size]).astype(np.int16)
                        img = self.image[coord_tar[1]:coord_tar[1] + self.patch_size,
                                         coord_tar[0]:coord_tar[0] + self.patch_size, :]
                        if img.any():
                            # cv2.putText(orig_img, "1", coord, cv2.FONT_HERSHEY_SIMPLEX, 20, color=(0, 0, 0), thickness=10)
                            self.patches_bag_list.append(coord_tar)
                            # cv2.imwrite(f"{result_path}/{single_name}/infer/{coord}_{j}_{k}.jpg", img)
        # cv2.imwrite(f"{result_path}/{single_name}/{single_name}_img.jpg", orig_img)

    def __len__(self):
        return len(self.patches_bag_list)

    def __getitem__(self, idx):
        coord_tar = self.patches_bag_list[idx]

        img = self.image[coord_tar[1]:coord_tar[1] + self.patch_size, coord_tar[0]:coord_tar[0] + self.patch_size, :]

        img = cv2.resize(img, (224, 224))
        img_tar = self.transform(img)

        return img_tar, coord_tar

