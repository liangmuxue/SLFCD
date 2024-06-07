import os
import sys
import logging
import argparse

import numpy as np
import openslide
import cv2
import json

from utils.constance import get_label_with_group_code

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tumor mask of tumor-WSI and '
                                             'save it in npy format')
parser.add_argument('--wsi_path', default=None, type=str, help='Path to the WSI file')
parser.add_argument('--level', default=1, type=int, help='at which WSI level'
                                                         ' to obtain the mask, default 1')


def run(wsi_path, npy_path, json_path, level=0):
    for json_file in os.listdir(json_path):
        single_name = json_file.split(".")[0]
        json_file_path = os.path.join(json_path, json_file)
        single_name = json_file.split(".")[0]
        wsi_file_path = os.path.join(wsi_path, single_name + ".svs")
        try:
            mask_tumor = get_mask_tumor(wsi_file_path, json_file_path, level=level)
            if mask_tumor is None:
                continue
            npy_file = os.path.join(npy_path, single_name + ".npy")
            np.save(npy_file, mask_tumor)
            print("process {} ok".format(json_file))
        except Exception as e:
            print("process json file fail, ignore:{}".format(single_name))
            continue


def get_mask_tumor(wsi_file_path, json_file_path, level=0):
    slide = openslide.OpenSlide(wsi_file_path)
    if len(slide.level_dimensions) <= level:
        print("no level for {},ignore:".format(wsi_file_path))
        return None

    # 获取指定层级的图像尺寸
    w, h = slide.level_dimensions[level]
    # 初始化一个全零的掩膜数组
    mask_tumor = np.zeros((h, w))

    # 获取指定层级的缩放比例
    scale = slide.level_downsamples[level]
    with open(json_file_path) as f:
        dicts = json.load(f)
    # 获取 JSON 数据中标记为正样本的多边形区域
    tumor_polygons = dicts['positive']
    # 遍历所有肿瘤多边形区域
    for tumor_polygon in tumor_polygons:
        group_name = tumor_polygon["group_name"]
        # 获取多边形顶点坐标并根据缩放比例进行缩放
        vertices = np.array(tumor_polygon["vertices"]) / scale
        vertices = vertices.astype(np.int32)
        # 不同组的不同掩码标志
        code = get_label_with_group_code(group_name)["code"]
        # 根据肿瘤分组名称获取对应的标签代码
        # 多个多边形填充  (code, code, code)
        cv2.fillPoly(mask_tumor, [vertices], (255, 255, 255))
    mask_tumor = mask_tumor.astype(np.uint8)
    return mask_tumor


def main(args):
    logging.basicConfig(level=logging.INFO)
    # file_path = "/home/bavon/datasets/wsi/lsil"
    file_path = "/home/bavon/datasets/wsi/hsil"
    file_path = args.wsi_path
    level = args.level
    # file_path = "/home/bavon/datasets/wsi/normal"
    wsi_path = "{}/data".format(file_path)
    npy_path = "{}/tumor_mask_level{}".format(file_path, level)
    if not os.path.exists(npy_path):
        os.mkdir(npy_path)
    json_path = "{}/json".format(file_path)
    run(wsi_path, npy_path, json_path, level=level)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
