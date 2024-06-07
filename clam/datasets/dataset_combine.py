from __future__ import print_function, division
import os
import numpy as np
import cv2
import openslide
from torch.utils.data import Dataset
from torchvision import transforms
import h5py

from utils.constance import get_label_cate_num


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
    def __init__(self, file_path, wsi_path, mask_path, patch_path=None, split_data=None, custom_downsample=1,
                 target_patch_size=-1, patch_level=0, patch_size=256, mode="lsil", transform=None, work_type="train", 
                 save_slide=False):
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
        self.transform = transform

        patches_bag_list = []
        pathces_len = 0
        labels = []
        # 处理非标注数据，遍历所有h5数据，存储相关坐标
        for svs_file in split_data:
            single_name = svs_file.split(".")[0]
            patch_file = os.path.join(file_path, patch_path, single_name + ".h5")
            wsi_file = os.path.join(file_path, "data", svs_file)
            wsi_data = openslide.open_slide(wsi_file)
            scale = wsi_data.level_downsamples[patch_level]
            with h5py.File(patch_file, "r") as f:
                patch_coords = np.array(f['coords'])
                # 存储标注框信息，形状为: (patch数量,最大长度,3)
                annotations = np.array(f["annotations"])
                label_data = np.array(f["label_data"])
                # 对应每个patch，包含的标注框实际数量
                boxes_len = np.array(f["boxes_len"])

                # 累计总记录数
                pathces_len += boxes_len.sum()
                if target_patch_size > 0:
                    # 如果目标分片尺寸大于0，需要对应处理，一般用不到
                    target_patch_size = (target_patch_size,) * 2
                elif custom_downsample > 1:
                    target_patch_size = (patch_size // custom_downsample,) * 2

                has_anno_flag = np.zeros(patch_coords.shape[0])
                # 对每个patch坐标进行处理，包括设置缩放等
                for i in range(patch_coords.shape[0]):
                    # 根据patch对应label数据，设置类别
                    single_label = label_data[i]
                    coord = patch_coords[i]
                    if single_label > 0:
                        # 病症
                        patches_type = "tumor"
                        # 设置当前切片是否包含标注
                        has_anno_flag[i] = 1
                    else:
                        patches_type = "normal"
                        # 如果当前属于未标注，则加入数据集合
                        patches_bag = {"type": patches_type, "label": 0, "scale": scale, "name": single_name,
                                       "patch_level": patch_level, "coord": (np.array(coord) / scale).astype(np.int16)}
                        patches_bag["anno_coord"] = patches_bag["coord"]
                        labels.append(0)
                        patches_bag_list.append(patches_bag)

                    # 遍历当前patch下的所有标注候选框，展开为对应记录
                    box_len = boxes_len[i]
                    for j in range(box_len):
                        patches_bag = {"type": patches_type}
                        box = annotations[i, j]
                        # 标签代码转为标签序号
                        patches_bag["label"] = get_label_cate_num(box[-1], mode=mode)
                        patches_bag["scale"] = scale
                        patches_bag["name"] = single_name
                        patches_bag["coord"] = (np.array(coord) / scale).astype(np.int16)
                        patches_bag["patch_level"] = patch_level

                        # 实际坐标
                        patches_bag["anno_coord"] = box[:-1].astype(np.int16)
                        patches_bag_list.append(patches_bag)
                        labels.append(patches_bag["label"])
                        if save_slide:
                            # 保存病症图片
                            img = wsi_data.read_region((patches_bag["anno_coord"] * scale).astype(np.int16), self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
                            if not os.path.exists(os.path.join(self.mask_path, single_name)):
                                os.makedirs(os.path.join(self.mask_path, single_name))
                            img.save(os.path.join(self.mask_path, single_name, f"{j}.png"))

        labels = np.array(labels)
        no_anno_index = np.where(labels == 0)[0]
        has_anno_index = np.where(labels > 0)[0]
        # 削减非标注样本数量，以达到数据规模平衡
        if work_type == "train":
            rate = 4
        else:
            rate = 4
        # 削减未标注区域数量，和已标注区域数量保持一致比例
        patches_bag_list = np.array(patches_bag_list)
        keep_no_anno_num = rate * has_anno_index.shape[0]
        if keep_no_anno_num < no_anno_index.shape[0]:
            target_pathces_idx = np.random.randint(0, no_anno_index.shape[0] - 1, (keep_no_anno_num,))
            no_anno_bag_list = patches_bag_list[target_pathces_idx].tolist()
            self.patches_bag_list = patches_bag_list[has_anno_index].tolist() + no_anno_bag_list
        else:
            self.patches_bag_list = patches_bag_list.tolist()
        self.pathces_total_len = len(self.patches_bag_list)

    def __len__(self):
        return self.pathces_total_len

    def __getitem__(self, idx):
        item = self.patches_bag_list[idx]
        name, coord, scale, label = item["name"], item["anno_coord"], item["scale"], item["label"]

        # 读取wsi文件并生成图像数据
        wsi_file = os.path.join(self.file_path, "data", name + ".svs")
        wsi = openslide.open_slide(wsi_file)

        coord = (coord * scale).astype(np.int16)
        img = wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if self.transform is not None:
            img_tar = self.transform(img)
        else:
            img_tar = img
        # 返回数据 训练图片/图片标签/图片信息/原始裁剪图片
        return img_tar, label, item, img


class Whole_Slide_Det(Whole_Slide_Bag_COMBINE):
    """针对目标检测模式的WSI数据集"""
    def __init__(self, file_path, wsi_path, mask_path, patch_path=None, split_data=None, custom_downsample=1,
                 target_patch_size=-1, patch_level=0, patch_size=256, mode="lsil", work_type="train", transform=None):
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
        self.transform = transform

        wsi_data = {}
        patches_bag_list = []
        patches_tumor_patch_file_list = []
        pathces_len = 0
        anno_flags = []
        # 处理非标注数据，遍历所有h5数据，存储相关坐标
        for svs_file in split_data:
            single_name = svs_file.split(".")[0]
            patch_file = os.path.join(file_path, patch_path, single_name + ".h5")
            wsi_file = os.path.join(file_path, "data", svs_file)
            wsi_data = openslide.open_slide(wsi_file)
            scale = wsi_data.level_downsamples[patch_level]
            with h5py.File(patch_file, "r") as f:
                single_name = os.path.basename(patch_file).split(".")[0]
                patch_coords = np.array(f['coords'])
                patch_level = f['coords'].attrs['patch_level']
                patch_size = f['coords'].attrs['patch_size']
                bboxes = np.array(f["annotations"])
                label_data = f["annotations"].attrs['label_data']
                boxes_len = f["annotations"].attrs['boxes_len']

                # 累计总记录数
                pathces_len += patch_coords.shape[0]
                if target_patch_size > 0:
                    # 如果目标分片尺寸大于0，需要对应处理，一般用不到
                    target_patch_size = (target_patch_size,) * 2
                elif custom_downsample > 1:
                    target_patch_size = (patch_size // custom_downsample,) * 2

                has_anno_flag = np.zeros(patch_coords.shape[0])
                # 对每个patch坐标进行处理，包括设置缩放等
                for i in range(patch_coords.shape[0]):
                    patches_bag = {"name": single_name, "scale": scale, "type": "normal"}
                    # 根据patch对应label数据，设置类别
                    single_label = label_data[i]
                    if single_label > 0:
                        patches_bag["type"] = "tumor"
                        # 设置当前切片是否包含标注
                        has_anno_flag[i] = 1
                    coord = patch_coords[i]
                    # 结构化属性
                    patches_bag["label"] = single_label
                    patches_bag["name"] = single_name
                    patches_bag["coord"] = np.array(coord) / scale
                    patches_bag["coord"] = patches_bag["coord"].astype(np.int16)
                    patches_bag["patch_level"] = patch_level
                    # 标注候选框,截取为实际长度
                    box_len = boxes_len[i]
                    patches_bag["bboxes"] = bboxes[i, :box_len, :].astype(np.int16)
                    # 变换为针对当前patch区域的坐标
                    if patches_bag["bboxes"].shape[0] > 0:
                        patches_bag["bboxes"][:, 0] = patches_bag["bboxes"][:, 0] - patches_bag["coord"][0]
                        patches_bag["bboxes"][:, 1] = patches_bag["bboxes"][:, 1] - patches_bag["coord"][1]
                        patches_bag["bboxes"][:, 2] = patches_bag["bboxes"][:, 2] - patches_bag["coord"][0]
                        patches_bag["bboxes"][:, 3] = patches_bag["bboxes"][:, 3] - patches_bag["coord"][1]
                        # 标签代码转为标签序号
                        for j in range(patches_bag["bboxes"].shape[0]):
                            patches_bag["bboxes"][j, 4] = get_label_cate_num(patches_bag["bboxes"][j, 4])
                            # 设置为同一类别
                            patches_bag["bboxes"][j, 4] = 0
                    patches_bag_list.append(patches_bag)
            anno_flags.append(has_anno_flag)

        anno_flags = np.concatenate(anno_flags)
        no_anno_index = np.where(anno_flags == 0)[0]
        has_anno_index = np.where(anno_flags == 1)[0]
        # 削减非标注样本数量，以达到数据规模平衡
        if work_type == "train":
            rate = 3
        else:
            rate = 5
        # 削减未标注区域数量，和已标注区域数量保持一致比例
        patches_bag_list = np.array(patches_bag_list)
        keep_no_anno_num = rate * has_anno_index.shape[0]
        if keep_no_anno_num < no_anno_index.shape[0]:
            target_pathces_idx = np.random.randint(0, no_anno_index.shape[0] - 1, (keep_no_anno_num,))
            no_anno_bag_list = patches_bag_list[target_pathces_idx].tolist()
            self.patches_bag_list = no_anno_bag_list + patches_bag_list[has_anno_index].tolist()
        else:
            self.patches_bag_list = patches_bag_list.tolist()
        self.pathces_total_len = len(self.patches_bag_list)

    def __getitem__(self, idx):

        item = self.patches_bag_list[idx]
        name = item["name"]
        coord = item["coord"]
        scale = item["scale"]
        annot = item["bboxes"]
        # 读取wsi文件并生成图像数据
        wsi_file = os.path.join(self.file_path, "data", name + ".svs")
        wsi = openslide.open_slide(wsi_file)
        coord_ori = (coord * scale).astype(np.int16)
        img = wsi.read_region(coord_ori, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # 返回数据包括图像数据以及标注候选框
        sample = {'img': img, 'annot': annot, "item": item}

        if self.transform:
            sample = self.transform(sample)
        return sample
