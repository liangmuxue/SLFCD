from __future__ import print_function, division
import os
import time

import numpy as np
import cv2
import openslide
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
from tqdm import tqdm

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
                 target_patch_size=-1, patch_level=0, patch_size=256, mode="lsil", transform=None, work_type="train"):
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
        for svs_file in tqdm(split_data, total=len(split_data), desc=f"process data {work_type}"):
            single_name = svs_file.split(".")[0]
            patch_file = os.path.join(file_path, patch_path, single_name + ".h5")
            wsi_file = os.path.join(file_path, "data", svs_file)
            wsi_data = openslide.open_slide(wsi_file)
            scale = wsi_data.level_downsamples[patch_level]
            with h5py.File(patch_file, "r") as f:
                patch_coords = np.array(f['coords'])
                # 存储标注框信息，形状为: (patch数量,最大长度,3)  # 488 489
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
                orig_img = wsi_data.read_region((0, 0), self.patch_level,
                                                wsi_data.level_dimensions[self.patch_level]).convert("RGB")
                orig_img = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
                orig_img_save = orig_img.copy()

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
                        if not os.path.exists(os.path.join(self.mask_path, single_name, "1")):
                            os.makedirs(os.path.join(self.mask_path, single_name, "1"))
                        # img = orig_img_save[coord[1]: coord[1] + self.patch_size,
                        #                     coord[0]: coord[0] + self.patch_size]
                        # cv2.imwrite(os.path.join(self.mask_path, single_name, "1", f"{coord}.jpg"), img)
                        coord = (np.array(coord) / scale).astype(np.int16)
                        cv2.putText(orig_img, "1",
                                    (int(coord[0] + patch_size / 2), int(coord[1] + patch_size / 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 0, 0), thickness=10)
                    else:
                        patches_type = "normal"
                        # 如果当前属于未标注，则加入数据集合
                        patches_bag = {"type": patches_type, "label": 0, "scale": scale, "name": single_name,
                                       "patch_level": patch_level, "coord": (np.array(coord) / scale).astype(np.int16)}
                        patches_bag["anno_coord"] = patches_bag["coord"]
                        labels.append(0)
                        patches_bag_list.append(patches_bag)
                        if not os.path.exists(os.path.join(self.mask_path, single_name, "0")):
                            os.makedirs(os.path.join(self.mask_path, single_name, "0"))
                        # img = orig_img_save[coord[1]: coord[1] + self.patch_size,
                        #                     coord[0]: coord[0] + self.patch_size]
                        # cv2.imwrite(os.path.join(self.mask_path, single_name, "0", f"{coord}.jpg"), img)
                        coord = (np.array(coord) / scale).astype(np.int16)
                        cv2.putText(orig_img, "0",
                                    (int(coord[0] + patch_size / 2), int(coord[1] + patch_size / 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 255), thickness=10)

                    # 遍历当前patch下的所有标注候选框，展开为对应记录
                    box_len = boxes_len[i]
                    for j in range(box_len):
                        patches_bag = {"type": patches_type}
                        box = annotations[i, j]
                        # 标签代码转为标签序号
                        patches_bag["label"] = get_label_cate_num(box[-2], mode=mode)
                        patches_bag["scale"] = scale
                        patches_bag["name"] = single_name
                        patches_bag["coord"] = (np.array(coord) / scale).astype(np.int16)
                        patches_bag["patch_level"] = patch_level

                        # 实际坐标
                        patches_bag["anno_coord"] = box[:-2].astype(np.int16)
                        patches_bag_list.append(patches_bag)
                        labels.append(patches_bag["label"])

                cv2.imwrite(f"{self.mask_path}/{single_name}/{single_name}.jpg", orig_img)

                # patches_bag_list = patches_bag_list[:int(len(patches_bag_list) / 2)]
                # labels = labels[:int(len(labels) / 2)]
                # end_time = time.time()
                # print(f"{svs_file} process time: {end_time - start_time} sed")

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


class Whole_Slide_Bag_COMBINE_all(Dataset):
    """
    Custom slide dataset,use multiple wsi file,in which has multiple patches
    """
    def __init__(self, file_path, wsi_path, mask_path, split_data=None, image_size=224,
                 patch_level=0, transform=None, work_type="train"):
        self.patch_level = patch_level
        self.file_path = file_path
        self.wsi_path = wsi_path
        self.mask_path = mask_path
        self.transform = transform
        self.image_size = image_size

        # 方法1
        # self.coords = []
        # for svs_file in tqdm(split_data, total=len(split_data), desc=f"process data {work_type}"):
        #     single_name = svs_file.split(".")[0]
        #     patch_file = os.path.join(file_path, patch_path, single_name + ".h5")
        #     wsi_file = os.path.join(file_path, "data", svs_file)
        #     wsi_data = openslide.open_slide(wsi_file)
        #
        #     if not os.path.exists(f'{self.mask_path}/{single_name}'):
        #         os.makedirs(f'{self.mask_path}/{single_name}', exist_ok=True)
        #
        #     with h5py.File(patch_file, "r") as f:
        #         positive_coords = np.array(f['positive'])
        #         negative_coords = np.array(f["negative"])
        #
        #         image = np.array(wsi_data.read_region((0, 0), self.patch_level, wsi_data.level_dimensions[self.patch_level]))
        #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #         image_copy = image.copy()
        #         for coord in positive_coords:
        #             cv2.rectangle(image, coord[:2], coord[2:], (0, 0, 0), 20, 2)
        #             img = image_copy[coord[1]:coord[3], coord[0]:coord[2], :]
        #             cv2.imwrite(f"{self.mask_path}/{single_name}/{coord}.jpg", img)
        #             self.coords.append({"name": single_name, 'anno_coord': coord, 'label': 1,
        #                                 'img_path': f"{self.mask_path}/{single_name}/{coord}.jpg"})
        #
        #         for coord in negative_coords:
        #             cv2.rectangle(image, coord[:2], coord[2:], (0, 255, 0), 20, 2)
        #             img = image_copy[coord[1]:coord[3], coord[0]:coord[2], :]
        #             cv2.imwrite(f"{self.mask_path}/{single_name}/{coord}.jpg", img)
        #             self.coords.append({"name": single_name, 'anno_coord': coord, 'label': 0,
        #                                 'img_path': f"{self.mask_path}/{single_name}/{coord}.jpg"})
        #         cv2.imwrite(f"{self.mask_path}/{single_name}/{single_name}.jpg", image)

        # 方法2
        self.all_files = []
        positive_num, negative_num = 0, 0
        for svs_file in tqdm(split_data, total=len(split_data), desc=f"process data {work_type}"):
            single_name = svs_file.split(".")[0]
            positive_file = os.path.join(file_path, 'mask', single_name, 'positive')
            negative_file = os.path.join(file_path, 'mask', single_name, 'negative')
            for i in os.listdir(positive_file):
                positive_num += 1
                self.all_files.append({"name": single_name, 'label': 1, 'img_path': os.path.join(positive_file, i)})
            for i in os.listdir(negative_file):
                negative_num += 1
                self.all_files.append({"name": single_name, 'label': 0, 'img_path': os.path.join(negative_file, i)})
        print(f"{work_type} data procefss num -> positive {positive_num} & negative {negative_num}")
                

    def letterbox(self, im, new_shape=(640, 640), color=(0, 0, 0), auto=False, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        item = self.all_files[idx]

        # 方法1
        # name, anno_coord, label, img_path = item["name"], item["anno_coord"], item["label"], item["img_path"]
        # img = cv2.imread(img_path)
        # if label == 1:
        #     img = self.letterbox(img, new_shape=self.crop_size)

        # 方法2
        label, img_path = item["label"], item["img_path"]
        img = cv2.imread(img_path)
        if self.transform is not None:
            img = cv2.resize(img, (self.image_size, self.image_size))
            img_tar = self.transform(img)
        else:
            img_tar = img
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
