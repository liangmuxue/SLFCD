# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'D:\BaiduNetdiskDownload\openslide-bin-4.0.0.3-windows-x64\bin'

import os

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

import os
import argparse
from shutil import copyfile
import pandas as pd
import numpy as np
import json
import openslide
import cv2
import h5py
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.constance import get_label_with_group_code, get_combine_label_with_type, get_tumor_label_cate
from utils.cv_utils import rect_overlap
from custom.tumor_mask import get_mask_tumor
from visdom import Visdom

# viz_debug = Visdom(env="debug")


def align_xml_svs(file_path):
    """Solving the problem of inconsistent naming between SVS and XML"""
    wsi_path = file_path + "/data"
    ori_xml_path = file_path + "/xml_ori"
    target_xml_path = file_path + "/xml"

    if not os.path.exists(target_xml_path):
        os.mkdir(target_xml_path)

    print("align_xml_svs -> wsi_path: ", wsi_path)
    print("align_xml_svs -> ori_xml_path: ", ori_xml_path)
    print("align_xml_svs -> target_xml_path: ", target_xml_path)

    for wsi_file in os.listdir(wsi_path):
        if not wsi_file.endswith(".svs"):
            continue
        single_name = wsi_file.split(".")[0]
        if "-" in single_name and False:
            xml_single_name = single_name.split("-")[0]
        else:
            xml_single_name = single_name
        xml_single_name = xml_single_name + ".xml"
        ori_xml_file = os.path.join(ori_xml_path, xml_single_name)
        tar_xml_file = os.path.join(target_xml_path, single_name + ".xml")
        try:
            copyfile(ori_xml_file, tar_xml_file)
        except Exception as e:
            print("copyfile fail,source:{} and target:{}".format(ori_xml_file, tar_xml_file), e)


def build_data_csv(file_path, split_rate=0.7):
    """build train and valid list to csv"""
    list_file = os.path.join(file_path, "process_list_autogen.csv")
    file_list = pd.read_csv(list_file)

    file_list_shuffled = file_list.sample(frac=1, random_state=None)
    total_file_number = file_list_shuffled.shape[0]

    train_number = int(total_file_number * split_rate)
    train_file_path = file_path + "/train.csv"
    valid_file_path = file_path + "/valid.csv"

    list_train, list_valid = [], []
    for i, wsi_files in enumerate(file_list_shuffled["slide_id"].values):
        single_name = wsi_files.split(".")[0]
        wsi_file = single_name + ".svs"
        if i < train_number:
            list_train.append([wsi_file, 1])
        else:
            list_valid.append([wsi_file, 1])

    train_df = pd.DataFrame(np.array(list_train), columns=['slide_id', 'label'])
    valid_df = pd.DataFrame(np.array(list_valid), columns=['slide_id', 'label'])
    train_df.to_csv(train_file_path, index=False, sep=',')
    valid_df.to_csv(valid_file_path, index=False, sep=',')
    print("split successful: ", train_file_path, ' and ', valid_file_path)


def crop_with_annotation(file_path, level=1):
    """Crop image from WSI refer to annotation"""
    crop_img_path = file_path + "/crop_img"
    patch_path = file_path + "/patches_level{}".format(level)
    wsi_path = file_path + "/data"
    json_path = file_path + "/json"
    total_file_number = len(os.listdir(json_path))

    if not os.path.exists(crop_img_path):
        os.makedirs(crop_img_path)
    if not os.path.exists(patch_path):
        os.makedirs(patch_path)
    print("crop_img_path: ", crop_img_path)
    print("patch_path: ", patch_path)
    print("wsi_path: ", wsi_path)
    print("json_path: ", json_path)
    print("total_file_number: ", total_file_number)

    # 一条一条的处理文件
    for i, json_file in tqdm(enumerate(os.listdir(json_path)), total=total_file_number, desc="crop with annotation"):
        json_file_path = os.path.join(json_path, json_file)
        single_name = json_file.split(".")[0]
        wsi_file = os.path.join(wsi_path, single_name + ".svs")

        # 缩放比例
        wsi = openslide.open_slide(wsi_file)
        scale = wsi.level_downsamples[level]
        with open(json_file_path, 'r') as jf:
            anno_data = json.load(jf)

        # 将不规则批注转换为矩形
        region_data, label_data = [], []
        for i, anno_item in enumerate(anno_data["positive"]):
            vertices = np.array(anno_item["vertices"])
            group_name = anno_item["group_name"]
            label = get_label_with_group_code(group_name)['code']
            label_data.append(label)

            # 获取到不规则区域的最小矩形框
            x_min = vertices[:, 0].min()
            x_max = vertices[:, 0].max()
            y_min = vertices[:, 1].min()
            y_max = vertices[:, 1].max()

            # 缩放到普通尺寸（层级图）
            region_size = (int((x_max - x_min) / scale), int((y_max - y_min) / scale))
            xywh = [x_min, y_min, region_size[0], region_size[1]]
            region_data.append(xywh)

            crop_img = np.array(wsi.read_region((x_min, y_min), level, region_size).convert("RGB"))
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
            img_file_name = "{}_{}-{}.jpg".format(single_name, i, label)
            img_file_path = os.path.join(crop_img_path, img_file_name)
            cv2.imwrite(img_file_path, crop_img)

        # 将裁剪的区域保存为h5
        patch_file_path = os.path.join(patch_path, single_name + ".h5")
        with h5py.File(patch_file_path, "a") as f:
            if "crop_region" in f:
                del f["crop_region"]
            f.create_dataset('crop_region', data=np.array(region_data))
            # 每一个剪裁区域都对应一个标签
            f['crop_region'].attrs['label_data'] = label_data


def patch_anno_img_lsil(xywh, mask_threhold, overlap_rate, is_edge_judge=False, patch_size=256,
                        mask_data=None, scale=4, file_path=None, label=1, file_name=None, index=0, level=1, wsi=None):
    """Crop annotation image with patch size"""

    tumor_patch_path = os.path.join(file_path, "tumor_patch_img")

    start_x, start_y, width, height = xywh
    start_x = start_x / scale
    start_y = start_y / scale
    end_x = start_x + width
    end_y = start_y + height

    def write_to_disk(patch_region, row=0, column=0):
        tumor_patch_file_path = os.path.join(tumor_patch_path,
                                             "{}/origin/{}_{}{}.jpg".format(label, file_name, row, column))
        top_left = (int(patch_region[0] * scale), int(patch_region[2] * scale))
        img_data = wsi.read_region(top_left, level, (patch_size, patch_size)).convert('RGB')
        img_data = cv2.cvtColor(np.array(img_data), cv2.COLOR_RGB2BGR)
        cv2.imwrite(tumor_patch_file_path, img_data)

    # Ignor small image
    if not is_edge_judge:
        if width < patch_size or height < patch_size:
            return None
        # ext_w = patch_size - width
        # ext_h = patch_size - height
        # region = [int(start_x - ext_w/2),int(end_x + ext_w/2),int(start_y - ext_h/2),int(end_y + ext_h/2)]
        # write_to_disk(region)
        # return np.expand_dims(np.array(region),axis=0)

    def step_crop(row_index, column_index, overlap_rate):
        """Overlap crop image,Stopping crop when cross the border refer to patch length"""
        x_start = int(start_x + patch_size * column_index * overlap_rate)
        x_end = x_start + patch_size
        y_start = int(start_y + patch_size * row_index * overlap_rate)
        y_end = y_start + patch_size

        if not is_edge_judge:
            if y_start > end_y:
                return None, -1
            if x_start > end_x:
                return None, 0

        patch_data = [x_start, x_end, y_start, y_end]
        return patch_data, 0

    row = 0
    patch_regions = []
    # Iterate rows and columns one.py by one.py,and crop image by patch size
    while True:
        column = 0
        while True:
            patch_region, flag = step_crop(row, column, overlap_rate)
            # If cross the width border, then switch to next row
            if patch_region is None:
                break
            # ReFilter with mask
            patch_masked = mask_data[patch_region[2]:patch_region[3], patch_region[0]:patch_region[1]]
            if (np.sum(patch_masked > 0) / (patch_size * patch_size)) > mask_threhold:
                patch_regions.append(patch_region)
                # Save to disk
                write_to_disk(patch_region, row=row, column=column)
                # viz_crop_patch(file_path,file_name,xywh,patch_region,viz=viz_debug)
            # else:
            #     viz_crop_patch(file_path,file_name,xywh,patch_region)
            column += 1
            # Cross the height border, break
        if flag == -1:
            break
        row += 1

    if len(patch_regions) > 0:
        patch_regions = np.stack(patch_regions)
    else:
        patch_regions = np.array([])
    return patch_regions


# hsil
def patch_anno_img(xywh, patch_size=256, mask_threhold=0.9, mask_data=None, scale=4, file_path=None, label=1,
                   file_name=None, index=0, level=1, wsi=None):
    """Crop annotation image with patch size"""

    tumor_patch_path = os.path.join(file_path, "tumor_patch_img")

    start_x, start_y, width, height = xywh
    start_x = start_x / scale
    start_y = start_y / scale
    end_x = start_x + width
    end_y = start_y + height

    def write_to_disk(patch_region, row=0, column=0):
        path = os.path.join(tumor_patch_path, "{}/origin".format(label))
        if not os.path.exists(path):
            os.makedirs(path)
        tumor_patch_file_path = os.path.join(path, "{}_{}{}.jpg".format(file_name, row, column))
        top_left = (int(patch_region[0] * scale), int(patch_region[2] * scale))
        img_data = wsi.read_region(top_left, level, (patch_size, patch_size)).convert('RGB')
        img_data = cv2.cvtColor(np.array(img_data), cv2.COLOR_RGB2BGR)
        cv2.imwrite(tumor_patch_file_path, img_data)

    def step_crop(row_index, column_index, overlap_rate=0.3):
        """Overlap crop image,Stopping crop when cross the border refer to patch length"""
        x_start = int(start_x + patch_size * column_index * overlap_rate)
        x_end = x_start + patch_size
        y_start = int(start_y + patch_size * row_index * overlap_rate)
        y_end = y_start + patch_size

        if y_start > end_y:
            return None, -1
        if x_start > end_x:
            return None, 0

        patch_data = [x_start, x_end, y_start, y_end]
        return patch_data, 0

    row = 0
    patch_regions = []
    # Iterate rows and columns one.py by one.py,and crop image by patch size
    while True:
        column = 0
        while True:
            patch_region, flag = step_crop(row, column)
            # If cross the width border, then switch to next row
            if patch_region is None:
                break
            # ReFilter with mask
            patch_masked = mask_data[patch_region[2]:patch_region[3], patch_region[0]:patch_region[1]]
            # 当掩码数量超过阈值时，说明属于对应标签的类别
            if (np.sum(patch_masked == label) / (patch_size * patch_size)) > mask_threhold:
                patch_regions.append(patch_region)
                # Save to disk
                write_to_disk(patch_region, row=row, column=column)
                # viz_crop_patch(file_path,file_name,xywh,patch_region,viz=viz_debug)
            # else:
            #     viz_crop_patch(file_path,file_name,xywh,patch_region)
            column += 1
            # Cross the height border, break
        if flag == -1:
            break
        row += 1

    if len(patch_regions) > 0:
        patch_regions = np.stack(patch_regions)
    else:
        patch_regions = np.array([])
    return patch_regions


def build_annotation_patches(file_path, level=1, patch_size=64, patch_slide_size=16, data_type=None,
                             mask_threhold=0.5, save_slide=True):
    """根据标注生成切片，针对图像识别模式"""
    xml_path = file_path + "/xml"

    # # 为每个patch创建一组滑动窗
    # def create_slide_windows(coord, patch_size, slide_time):
    #     # slide_window_patches = np.zeros((slide_time,slide_time,2))
    #     rows = np.linspace(coord[0], coord[0] + patch_size, num=slide_time)
    #     cols = np.linspace(coord[1], coord[1] + patch_size, num=slide_time)
    #     # 使用广播功能，以及水平拼接，生成二维数组
    #     xx, yy = np.meshgrid(rows, cols)
    #     slide_window_patches = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    #     slide_window_patches = slide_window_patches.reshape(slide_time, slide_time, 2)
    #     return slide_window_patches

    # for xml_file in tqdm(os.listdir(xml_path), total=len(os.listdir(xml_path)), desc='build_annotation_patches'):
    #     build_annotation_patches_single(file_path, xml_file, level=level, patch_size=patch_size,
    #                                     patch_slide_size=patch_slide_size, data_type=data_type,
    #                                     mask_threhold=mask_threhold)

    Parallel(n_jobs=8)(delayed(build_annotation_patches_single)(file_path, xml_file, level=level, patch_size=patch_size,
                                                                patch_slide_size=patch_slide_size, data_type=data_type,
                                                                mask_threhold=mask_threhold, save_slide=save_slide
                                                                ) for xml_file in os.listdir(xml_path))


def build_annotation_patches_single(file_path, xml_file, level=1, patch_size=64, patch_slide_size=16,
                                    data_type=None, mask_threhold=0.5, save_slide=True):
    patch_path = file_path + "/patches_level{}".format(level)
    wsi_path = file_path + "/data"

    # 每个patch的滑动次数
    slide_time = patch_size // patch_slide_size
    file_name = xml_file.split(".")[0]

    patch_file_path = os.path.join(patch_path, "{}.h5".format(file_name))
    wsi_file_path = os.path.join(wsi_path, file_name + ".svs")
    json_file_path = os.path.join(file_path, "json", file_name + ".json")

    wsi = openslide.open_slide(wsi_file_path)
    # 获取指定层级的缩放比例
    scale = wsi.level_downsamples[level]
    # 读取 h5 文件中的标准标注区域
    with h5py.File(patch_file_path, "a") as f:
        # 读取坐标数据 掩码全格式得到
        try:
            coords = np.array(f['coords'])
        except Exception as e:
            print("coords data None:{}".format(patch_file_path))
            return

        label_data = np.zeros((coords.shape[0]))
        anno_data = []

        # 获取肿瘤掩膜，人工标记的正标签（img）
        tumor_mask, mask_tumor_copy = get_mask_tumor(wsi_file_path, json_file_path, level=level)
        # 生成矩形候选框，人工标记的正标签 （label+xyxy）
        anno_regions = get_anno_box_datas(json_file_path, scale=scale)

        if not os.path.exists(file_path + "/slide_img/" + file_name):
            os.makedirs(file_path + "/slide_img/" + file_name)
        cv2.imwrite(file_path + f"/slide_img/{file_name}/{file_name}_mask.jpg", mask_tumor_copy)

        anno_region_items = np.array([item["region"] + [item["code"]] for item in anno_regions])
        max_box_len = 0

        orig_img = wsi.read_region((0, 0), level, wsi.level_dimensions[level]).convert("RGB")
        orig_img = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
        orig_img_copy1 = orig_img.copy()
        orig_img_copy2 = orig_img.copy()

        # 遍历每个剪裁区域，取得对应标注区域（非空白区域）
        shape = coords.shape
        print('patch_file_path: ', patch_file_path, "shape: ", shape)
        for i in range(shape[0]):
            anno_patches = []
            coord = coords[i]

            cv2.putText(orig_img_copy1, "1",
                        (int((coord[0] + patch_size / 2) / scale),
                         int((coord[1] + patch_size / 2) / scale)),
                         cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 0, 0), thickness=10)

            has_anno_flag = False
            # 使用滑动窗方式，对patch以及周边进行候选扫描
            # slide_window_patches = create_slide_windows(coord,patch_size,slide_time)
            # 通过两层嵌套循环遍历每个滑动窗口
            for k in range(slide_time):
                for j in range(slide_time):
                    # 计算当前滑动窗口的坐标
                    coord_cur = [int((coord[0] + k * patch_slide_size) / scale),
                                 int((coord[1] + j * patch_slide_size) / scale)]
                    # 判断当前滑动窗口是否包含标注区域，并获取标签
                    match_flag, label, conf, patch_masked_copy = judge_region_match_new(coord_cur, patch_size,
                                                                                        tumor_mask,
                                                                                        mask_tumor_copy,
                                                                                        anno_regions=anno_region_items,
                                                                                        mask_threhold=mask_threhold)
                    # 如果匹配，记录匹配的坐标点
                    if match_flag:
                        # 追加对应区域的标签信息
                        anno_patches.append(coord_cur + [label])
                        has_anno_flag = True

                        if save_slide:
                            if not os.path.exists(file_path + "/slide_img/" + file_name + "/mask"):
                                os.makedirs(file_path + "/slide_img/" + file_name + "/mask")
                            cv2.imwrite(
                                file_path + "/slide_img/" + file_name + "/mask/" + f"{i}_{j}_{conf:.2f}_org.jpg",
                                patch_masked_copy)

                        cv2.rectangle(orig_img_copy2, coord_cur,
                                      (coord_cur[0] + patch_size, coord_cur[1] + patch_size),
                                      (0, 0, 255), 4)

            if i % 1000 == 0:
                print("slide {} loop cont:{}".format(patch_file_path, i))

            # 计算取得当前文件对应的标注框最大值，用于后续数据补充对齐
            box_len = len(anno_patches)
            max_box_len = box_len if max_box_len < box_len else max_box_len
            # 当前坐标点包含标注，则在标签数据数组中标记
            if has_anno_flag:
                label_data[i] = 1
            anno_data.append(anno_patches)

        if not os.path.exists(file_path + "/slide_img/"):
            os.makedirs(file_path + "/slide_img/")
        cv2.imwrite(file_path + f"/slide_img/{file_name}/{file_name}_copy1.jpg", orig_img_copy1)
        cv2.imwrite(file_path + f"/slide_img/{file_name}/{file_name}_copy2.jpg", orig_img_copy2)

        boxes_len = []
        # 再次拼接为一个完整的数组,按照最大标注框数量构建数组
        bboxes_data_arr = np.zeros((coords.shape[0], max_box_len, 3))
        labels = None
        for i in range(coords.shape[0]):
            # 记录patch中的标注框数量
            boxes_len.append(len(anno_data[i]))
            # 如果当前坐标点有标注，则将其追加到标注框数据数组和标签数组中
            if len(anno_data[i]) > 0:
                bboxes_data_arr[i, :len(anno_data[i]), :] = np.array(anno_data[i])
                labels_item = np.array([item[-1] for item in anno_data[i]])
                if labels is None:
                    labels = labels_item
                else:
                    labels = np.concatenate((labels, labels_item))

        # 从新创建标注数据集
        if "annotations" in f:
            del f["annotations"]
        if "label_data" in f:
            del f["label_data"]
        if "boxes_len" in f:
            del f["boxes_len"]
            # 标注数据，以及相关信息
        f.create_dataset("annotations", data=bboxes_data_arr)
        f.create_dataset("label_data", data=label_data)
        f.create_dataset("boxes_len", data=boxes_len)


def judge_region_match(coord_cur, patch_size, tumor_mask, anno_regions=None, mask_threhold=0.5):
    """判断指定区域是否包含标注"""
    # 根据当前坐标和图像块大小从肿瘤掩膜中提取当前图像块区域
    patch_masked = tumor_mask[coord_cur[1]:coord_cur[1] + patch_size, coord_cur[0]:coord_cur[0] + patch_size]
    conf = np.sum(patch_masked > 0) / (patch_size * patch_size)
    # 如果标注面积占比超过了当前区域的一定比例，则属于包含标注
    if conf > mask_threhold:
        # 获取当前图像块区域中非零像素值的唯一值和它们的计数
        u, c = np.unique(patch_masked[patch_masked > 0], return_counts=True)
        # 选择计数最多的像素值作为标签
        label = u[c == c.max()]
        return True, label[0]

    if len(anno_regions.shape) < 2:
        return False, 0

    # 如果区域中包含完整的标注，也入选
    # 计算当前图像块的区域坐标
    patch_area = [coord_cur[0], coord_cur[1], coord_cur[0] + patch_size, coord_cur[1] + patch_size]
    # 检查 anno_regions 中是否有标注区域与当前图像块区域重叠
    match_idx = np.where((anno_regions[:, 0] > patch_area[0])
                         & (anno_regions[:, 1] > patch_area[1])
                         & (anno_regions[:, 2] < patch_area[2])
                         & (anno_regions[:, 3] < patch_area[3]))[0]
    if match_idx.shape[0] > 0:
        return True, anno_regions[match_idx[0]][-1]
    else:
        return False, 0


def judge_region_match_new(coord_cur, patch_size, tumor_mask, mask_tumor_copy, anno_regions=None, mask_threhold=0.5):
    """判断指定区域是否包含标注"""
    # 根据当前坐标和图像块大小从肿瘤掩膜中提取当前图像块区域
    patch_masked = tumor_mask[coord_cur[1]:coord_cur[1] + patch_size, coord_cur[0]:coord_cur[0] + patch_size]
    patch_masked_copy = mask_tumor_copy[coord_cur[1]:coord_cur[1] + patch_size, coord_cur[0]:coord_cur[0] + patch_size]
    conf = np.sum(patch_masked > 0) / (patch_size * patch_size)
    # 如果标注面积占比超过了当前区域的一定比例，则属于包含标注
    # if conf > mask_threhold:
    #     # 获取当前图像块区域中非零像素值的唯一值和它们的计数
    #     u, c = np.unique(patch_masked[patch_masked > 0], return_counts=True)
    #     # 选择计数最多的像素值作为标签
    #     label = u[c == c.max()]
    #     return True, label[0], conf, patch_masked_copy
    #
    # if len(anno_regions.shape) < 2:
    #     return False, 0, conf, patch_masked_copy

    # 如果区域中包含完整的标注，也入选
    # 计算当前图像块的区域坐标
    patch_area = [coord_cur[0], coord_cur[1], coord_cur[0] + patch_size, coord_cur[1] + patch_size]
    # 检查 anno_regions 中是否有标注区域与当前图像块区域重叠
    match_idx = np.where((anno_regions[:, 0] > patch_area[0])
                         & (anno_regions[:, 1] > patch_area[1])
                         & (anno_regions[:, 2] < patch_area[2])
                         & (anno_regions[:, 3] < patch_area[3]))[0]
    if match_idx.shape[0] > 0:
        return True, anno_regions[match_idx[0]][-1], conf, patch_masked_copy
    else:
        return False, 0, conf, patch_masked_copy



def build_annotation_patches_det(file_path, level=1, patch_size=64, mask_threhold=0.2, data_type=None):
    """根据标注生成切片，针对目标检测模式"""
    patch_path = file_path + "/patches_level{}".format(level)
    wsi_path = file_path + "/data"
    xml_path = file_path + "/xml"
    labels = get_tumor_label_cate(data_type)

    for xml_file in os.listdir(xml_path):
        file_name = xml_file.split(".")[0]
        patch_file = os.path.join(patch_path, "{}.h5".format(file_name))
        # if file_name!="80-CG23_15274_01":
        #     continue
        patch_file_path = os.path.join(patch_path, patch_file)
        wsi_file_path = os.path.join(wsi_path, file_name + ".svs")
        json_file_path = os.path.join(file_path, "json", file_name + ".json")
        wsi = openslide.open_slide(wsi_file_path)
        scale = wsi.level_downsamples[level]

        with h5py.File(patch_file_path, "a") as f:
            try:
                coords = np.array(f['coords'])
            except Exception as e:
                print("coords data None:{}".format(patch_file_path))
                continue
            crop_region = f['crop_region'][:]
            label_data = []
            patches_length = 0
            bboxes_data = []
            anno_regions = get_anno_box_datas(json_file_path, scale=scale)
            max_box_len = 0
            # 遍历每个剪裁区域
            for i in range(coords.shape[0]):
                coord = coords[i]
                # 取得对应坐标范围的图片
                top_left = (int(coord[0] * scale), int(coord[1] * scale))
                patch_region = (top_left[0], top_left[1], patch_size, patch_size)
                # 根据掩码数据，判断是否具备相关标注,并生成标注框(标注框有可能为多个)
                label, bboxes = build_patch_anno(patch_region, labels=labels, patch_size=patch_size,
                                                 mask_threhold=mask_threhold, anno_regions=anno_regions)
                box_len = len(bboxes)
                max_box_len = box_len if max_box_len < box_len else max_box_len
                label_data.append(label)
                bboxes_data.append(bboxes)

            boxes_len = []
            # 再次拼接为一个完整的数组,按照最大候选框数量构建数组
            bboxes_data_arr = np.zeros((len(bboxes_data), max_box_len, 5))
            for i in range(len(bboxes_data)):
                # 记录patch中的标注框数量
                boxes_len.append(len(bboxes_data[i]))
                if len(bboxes_data[i]) > 0:
                    bboxes_data_arr[i, :len(bboxes_data[i]), :] = np.array(bboxes_data[i])

            # 从新创建标注数据集
            if "annotations" in f:
                del f["annotations"]
            f.create_dataset("annotations", data=bboxes_data_arr)
            # 记录标签信息
            f["annotations"].attrs['label_data'] = label_data
            f["annotations"].attrs['boxes_len'] = boxes_len
            print("patch {} ok".format(file_name))


def get_anno_box_datas(json_filepath, scale=1):
    """取得标注数据,返回矩形候选框格式"""
    with open(json_filepath) as f:
        dicts = json.load(f)
    # 获取正样本的多边形区域
    tumor_polygons = dicts['positive']
    anno_regions = []
    # 遍历所有肿瘤多边形区域
    for tumor_polygon in tumor_polygons:
        group_name = tumor_polygon["group_name"]
        # 获取多边形顶点坐标并根据缩放比例进行缩放
        vertices = np.array(tumor_polygon["vertices"]) / scale
        vertices = vertices.astype(np.int32)
        # 根据肿瘤分组名称获取对应的标签代码
        code = get_label_with_group_code(group_name)["code"]
        # 存储当前标注框的标签和区域坐标 编码和矩形区域对应，box格式:xyxy
        region = {"code": code, "region": [vertices.min(axis=0)[0], vertices.min(axis=0)[1], vertices.max(axis=0)[0],
                                           vertices.max(axis=0)[1]]}
        anno_regions.append(region)
    return anno_regions


def build_patch_anno(patch_region, labels=None, mask_threhold=0.5, patch_size=64, anno_regions=None):
    """对于patch区域，根据是否有标注目标，生成相关数据"""
    # xywh转xyxy
    patch_region = [patch_region[0], patch_region[1], patch_region[0] + patch_region[2],
                    patch_region[1] + patch_region[3]]
    bboxes = []
    max_area = 0
    label = 0
    # if patch_region[0]>5000 and patch_region[1]>1500:
    #     print("ggg")
    # 循环所有标注区域，查找是否与当前patch区域相交
    for anno_region in anno_regions:
        region = anno_region["region"]
        code = anno_region["code"]
        # 取得相交区域
        ret = rect_overlap(region, patch_region)
        if len(ret) == 0:
            continue
        (X1, Y1, X2, Y2, area) = ret
        # 相交面积与标注本身的面积比例大于指定阈值，则认为是可用
        area_rate = area / (patch_size * patch_size)
        if area_rate < mask_threhold:
            continue
        # 根据最大面积取得整区域的标签
        if area > max_area:
            max_area = area
            label = code
        # 记录相交区域，以及对应标签
        box = [X1, Y1, X2, Y2, code]
        bboxes.append(box)
    return label, bboxes


def aug_annotation_patches(file_path, type, number, level=1):
    import Augmentor
    tumor_patch_path = os.path.join(file_path, "mask")
    # if type == 'lsil':
    #     labels = [4, 5]
    # elif type == 'hsil':
    #     labels = [1, 2, 3]
    # elif type == 'ais':
    #     labels = [7, 8, 9, 10]
    # for label in labels:
    for i in os.listdir(tumor_patch_path):
        img_path = os.path.join(tumor_patch_path, i, "positive")
        target_img_path = os.path.join(tumor_patch_path, i, "output")
        p = Augmentor.Pipeline(img_path, output_directory=target_img_path)

        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.sample(number)
        p.process()
        p.zoom_random(probability=1, percentage_area=0.8)
        p.sample(number)
        p.process()
        p.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
        p.sample(number)
        p.process()
        p.flip_left_right(probability=0.5)
        p.sample(number)
        p.process()
        p.flip_top_bottom(probability=0.5)
        p.sample(number)
        p.process()
        p.random_brightness(probability=1, min_factor=0.7, max_factor=1.2)
        p.sample(number)
        p.process()
        break


def filter_patches_exclude_anno(file_path, level=1, patch_size=256):
    """Remove annotation patches from origin coordinates"""
    patch_path = file_path + "/patches_level{}".format(level)
    wsi_path = file_path + "/data"
    for patch_file in os.listdir(patch_path):
        file_name = patch_file.split(".")[0]
        patch_file_path = os.path.join(patch_path, patch_file)
        wsi_file_path = os.path.join(wsi_path, file_name + ".svs")
        wsi = openslide.open_slide(wsi_file_path)
        scale = wsi.level_downsamples[level]
        mask_path = os.path.join(file_path, "tumor_mask_level{}".format(level))
        npy_file = os.path.join(mask_path, file_name + ".npy")
        if not os.path.exists(npy_file):
            print("file not exists:{}".format(npy_file))
            continue
        mask_data = np.load(npy_file)

        target_coords = []
        with h5py.File(patch_file_path, "a") as f:
            coords = f['coords'][:]
            for coord in coords:
                coord_x = int(coord[0] / scale)
                coord_y = int(coord[1] / scale)
                mask_data_item = mask_data[coord_y:coord_y + patch_size, coord_x:coord_x + patch_size]
                if np.sum(mask_data_item > 0) < 100:
                    target_coords.append(coord)
            attr_bak = {}
            for key in f['coords'].attrs:
                attr_bak[key] = f['coords'].attrs[key]
            del f['coords']
            f.create_dataset('coords', data=np.array(target_coords))
            for key in attr_bak:
                f["coords"].attrs[key] = attr_bak[key]

            print("patch {} ok".format(file_name))


def judge_patch_anno_lsil(coord, mask_thredhold, mask_data=None, scale=1, patch_size=64):
    """Judge if patch has annotation data"""
    coord_x = int(coord[0] / scale)
    coord_y = int(coord[1] / scale)
    mask_data_item = mask_data[coord_y:coord_y + patch_size, coord_x:coord_x + patch_size]
    t = mask_thredhold * patch_size * patch_size
    # No more mask data,then not has annotation data
    if np.sum(mask_data_item > 0) < t:
        return False
    return True


def judge_patch_anno(coord, mask_data=None, scale=1, patch_size=64, thres_hold=3):
    """Judge if patch has annotation data"""
    coord_x = int(coord[0] / scale)
    coord_y = int(coord[1] / scale)
    mask_data_item = mask_data[coord_y:coord_y + patch_size, coord_x:coord_x + patch_size]
    # No more mask data,then not has annotation data
    if np.sum(mask_data_item > 0) < thres_hold:
        return False
    return True


def label_patch_anno(coord, mask_data=None, scale=1, patch_size=64):
    """Judge if patch has annotation data"""
    coord_x = int(coord[0] / scale)
    coord_y = int(coord[1] / scale)
    mask_data_item = mask_data[coord_y:coord_y + patch_size, coord_x:coord_x + patch_size]

    unique_values, counts = np.unique(mask_data_item[mask_data_item > 0], return_counts=True)

    most_frequent_index = np.argmax(counts)

    most_frequent_value = unique_values[most_frequent_index]
    return most_frequent_value


def build_normal_patches_image(file_path, is_normal=False, level=1, patch_size=64):
    """Build images of normal region in wsi"""
    patch_path = file_path + "/patches_level{}".format(level)
    wsi_path = file_path + "/data"
    xml_path = file_path + "/xml"
    for file in os.listdir(patch_path):
        file_name = file.split(".")[0]
        # if relation xml config file not exists, then ignore
        if not is_normal and os.path.exists(os.path.join(xml_path, file_name + ".xml")):
            continue
        patch_file = file_name + ".h5"
        patch_file_path = os.path.join(patch_path, patch_file)
        wsi_file_path = os.path.join(wsi_path, file_name + ".svs")
        wsi = openslide.open_slide(wsi_file_path)
        scale = wsi.level_downsamples[level]
        mask_path = os.path.join(file_path, "tumor_mask_level{}".format(level))
        npy_file = os.path.join(mask_path, file_name + ".npy")
        if not is_normal:
            mask_data = np.load(npy_file)
        save_path = os.path.join(file_path, "tumor_patch_img/0", file_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("process file:{}".format(patch_file_path))
        with h5py.File(patch_file_path, "a") as f:
            if not "coords" in f:
                print("coords not in:{}".format(file_name))
                continue
            coords = f['coords'][:]
            for idx, coord in enumerate(coords):
                # Ignore annotation patches data
                if not is_normal:
                    if judge_patch_anno(coord, mask_data=mask_data, scale=scale, patch_size=patch_size):
                        continue
                crop_img = np.array(wsi.read_region(coord, level, (patch_size, patch_size)).convert("RGB"))
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                save_file_path = os.path.join(save_path, "{}.jpg".format(idx))
                print("save_file_path", save_file_path)
                cv2.imwrite(save_file_path, crop_img)
            print("write image ok:{}".format(file_name))


def combine_mul_dataset_csv(file_path, types):
    """Combine multiple tumor type csv,To: train,valid,test"""
    combine_train_split, combine_valid_split = None, None
    # 读取各自的文件夹里面的数据
    for type in types:
        # train
        type_csv_train = os.path.join(file_path, type, "train.csv")
        train_split = pd.read_csv(type_csv_train)
        train_split["label"] = get_combine_label_with_type(type, mode="ais")
        train_split.insert(train_split.shape[1], 'type', type)
        if combine_train_split is None:
            combine_train_split = train_split
        else:
            combine_train_split = pd.concat([combine_train_split, train_split])

        # val
        type_csv_valid = os.path.join(file_path, type, "valid.csv")
        valid_split = pd.read_csv(type_csv_valid)
        valid_split["label"] = get_combine_label_with_type(type, mode="ais")
        valid_split.insert(valid_split.shape[1], 'type', type)
        if combine_valid_split is None:
            combine_valid_split = valid_split
        else:
            combine_valid_split = pd.concat([combine_valid_split, valid_split])

    combine_train_split.reset_index(inplace=True)
    combine_valid_split.reset_index(inplace=True)
    combine_train_split['case_id'] = combine_train_split.index
    combine_valid_split['case_id'] = combine_valid_split.index

    # 划分数据集
    # 标签为1的样本
    label_1_df = combine_valid_split[combine_valid_split['label'] == 1]
    # 标签为0的样本
    label_0_df = combine_valid_split[combine_valid_split['label'] == 0]

    # 对标签为1的样本划分验证集和测试集
    val_label_1_df, test_label_1_df = train_test_split(label_1_df, test_size=0.3, random_state=42)
    # 对标签为0的样本划分验证集和测试集
    val_label_0_df, test_label_0_df = train_test_split(label_0_df, test_size=0.3, random_state=42)

    # 合并验证集
    combine_valid_sp = pd.concat([val_label_1_df, val_label_0_df])
    # 合并测试集
    combine_test_sp = pd.concat([test_label_1_df, test_label_0_df])

    # size = combine_valid_split.shape[0]
    # sp_size = int(size * 0.6)
    # combine_valid_sp = combine_valid_split.iloc[:sp_size]
    # combine_test_sp = combine_valid_split.iloc[sp_size:]

    output_path = os.path.join(file_path, "combine")
    train_file_path = os.path.join(output_path, "train.csv")
    valid_file_path = os.path.join(output_path, "valid.csv")
    test_file_path = os.path.join(output_path, "test.csv")
    combine_train_split.to_csv(train_file_path)
    combine_valid_sp.to_csv(valid_file_path)
    combine_test_sp.to_csv(test_file_path)

    print("train_file_path: ", train_file_path)
    print("valid_file_path: ", valid_file_path)
    print("test_file_path: ", test_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get tumor mask of tumor-WSI and save it in npy format')
    parser.add_argument('--source', default=r"/home/bavon/datasets/wsi", type=str, help='Path to the WSI file')
    parser.add_argument('--data_type', default='ais', type=str, help='数据类别：hsil lsil ais')
    parser.add_argument('--level', default=1, type=int, help='at which WSI level to obtain the mask, default 1')
    parser.add_argument('--patch_size', default=512, type=int, help='切片尺寸，默认64*64')
    parser.add_argument('--patch_slide_size', default=128, type=int, help='滑动窗距离')
    parser.add_argument('--save_slide', default=True, type=bool)

    args = parser.parse_args()
    print("args:", args)
    file_path = os.path.join(args.source, args.data_type)

    # align_xml_svs(file_path)
    # 划分第一阶段的训练集和测试集
    # build_data_csv(file_path)
    # 数据增强
    # aug_annotation_patches(file_path, 'ais', 2)

    # 添加标注区域
    # crop_with_annotation(file_path, level=args.level)

    # 添加滑块标注区域
    # build_annotation_patches(file_path, mask_threhold=0.15, level=args.level, data_type=args.data_type,
    #                          patch_size=args.patch_size, patch_slide_size=args.patch_slide_size,
    #                          save_slide=args.save_slide)

    types = ["ais", "normal"]
    file_path = args.source
    combine_mul_dataset_csv(file_path, types)

    # 目标检测模式
    # build_annotation_patches_det(file_path,mask_threhold=0.005,level=args.level,patch_size=args.patch_size,data_type=args.data_type)
    # aug_annotation_patches(file_path,'lsil',33)
    # filter_patches_exclude_anno(file_path,level=args.level)

    # is_normal = False
    # build_normal_patches_image(file_path,is_normal=is_normal)
    # types = ["lsil","normal"]
    print("process success!!!")


