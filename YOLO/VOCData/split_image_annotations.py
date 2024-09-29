OPENSLIDE_PATH = r'D:\BaiduNetdiskDownload\openslide-bin-4.0.0.3-windows-x64\bin'

import os

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

import argparse
import shutil
import openslide
import json
import os
import numpy as np
from PIL import ImageDraw
import xml.etree.ElementTree as ET
import cv2
import h5py


def get_anno_box_datas(json_filepath, scale):
    """
    取得标注数据,返回矩形候选框格式
    """
    with open(json_filepath) as f:
        dicts = json.load(f)
    tumor_polygons = dicts['positive']
    mask_tumor = np.zeros((shape[1], shape[0]), dtype=np.uint8)
    anno_regions = []
    for tumor_polygon in tumor_polygons:
        vertices = np.array(tumor_polygon["vertices"]) / scale
        vertices = vertices.astype(np.int32)
        # 最小矩形框
        min_matrix = [vertices.min(axis=0)[0], vertices.min(axis=0)[1],
                      vertices.max(axis=0)[0], vertices.max(axis=0)[1]]
        anno_regions.append(min_matrix)
        cv2.fillPoly(mask_tumor, [vertices], (255, 255, 255))
    return anno_regions, mask_tumor


def create_voc_xml(annotations, filename, path, savepath, width, height):
    """
    创建PASCAL VOC格式的XML文件
    """
    voc_root = ET.Element("annotation")

    ET.SubElement(voc_root, "folder").text = "images"
    ET.SubElement(voc_root, "filename").text = filename
    ET.SubElement(voc_root, "path").text = path

    source = ET.SubElement(voc_root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(voc_root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(voc_root, "segmented").text = "0"

    for annotation in annotations:
        obj = ET.SubElement(voc_root, "object")
        ET.SubElement(obj, "name").text = annotation['name']
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(annotation['xmin'])
        ET.SubElement(bndbox, "ymin").text = str(annotation['ymin'])
        ET.SubElement(bndbox, "xmax").text = str(annotation['xmax'])
        ET.SubElement(bndbox, "ymax").text = str(annotation['ymax'])

    tree = ET.ElementTree(voc_root)
    tree.write(savepath)


def remove_duplicates(rects):
    unique_rects = []
    seen = set()  # 使用集合来存储已经见过的矩形的唯一标识符

    for rect in rects:
        # 创建一个唯一标识符，例如通过连接所有坐标
        rect_id = (rect['xmin'], rect['ymin'], rect['xmax'], rect['ymax'])
        if rect_id not in seen:
            seen.add(rect_id)
            unique_rects.append(rect)

    return unique_rects


def is_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    # 检查x坐标是否有交集
    if x1 > x4 or x2 < x3:
        return False
    # 检查y坐标是否有交集
    if y1 > y4 or y2 < y3:
        return False
    # 如果以上条件都不满足，说明有交集
    return True


def split_image_and_annotations(name, image, h5_path, shape, anno_region, mask_tumor, img_size, patch_size, slide_size):
    slide_length = patch_size // slide_size
    with h5py.File(h5_path, "a") as f:
        coords = np.array(f['coords'])
        repeat_coord = []
        for coord in coords:
            coord = np.array(coord / scale).astype(int)
            for i in range(slide_length):
                for j in range(slide_length):
                    # 计算切分区域的坐标
                    coord_tar = np.array([coord[0] + i * slide_size, coord[1] + j * slide_size]).astype(np.int16)
                    coord_tar = [coord_tar[0], coord_tar[1], coord_tar[0] + img_size, coord_tar[1] + img_size]
                    if coord_tar[2] < shape[0] and coord_tar[3] < shape[1]:
                        if coord_tar not in repeat_coord:
                            repeat_coord.append(coord_tar)

                            if f'{name[:-4]}_{coord_tar}_1_1' in ais:
                                continue

                            x1, y1, x2, y2 = coord_tar

                            # 创建新的图像
                            new_image = image.crop((x1, y1, x2, y2))
                            draw_mask_tumor = mask_tumor.copy()
                            mask_image = draw_mask_tumor[y1: y2, x1: x2]

                            # 处理标注
                            new_annotations = []
                            flag = False
                            for bndbox in anno_region:
                                # 读取原始坐标
                                x_min, y_min, x_max, y_max = bndbox

                                # 判断是否有交集在原图上
                                if is_intersect(x1, y1, x2, y2, x_min, y_min, x_max, y_max):
                                    # 查找轮廓
                                    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    # 遍历轮廓，找到所有白色区域的矩形框
                                    for contour in contours:
                                        # 获取轮廓的边界框
                                        x, y, w, h = cv2.boundingRect(contour)
                                        new_x_min, new_y_min = x, y
                                        new_x_max, new_y_max = x + w, y + h
                                        print(f"白色区域的矩形框坐标: (x={x}, y={y}, width={w}, height={h})")

                                        # 全包括
                                        if x1 <= x_min and y1 <= y_min and x2 >= x_max and y2 >= y_max:
                                            new_annotations.append({
                                                'name': 'lsil',
                                                'xmin': str(new_x_min),
                                                'ymin': str(new_y_min),
                                                'xmax': str(new_x_max),
                                                'ymax': str(new_y_max)
                                            })
                                            continue

                                        # 面积太小
                                        if (new_x_max - new_x_min)/(new_y_max - new_y_min) < 0.4 or \
                                                (new_y_max - new_y_min)/(new_x_max - new_x_min) < 0.4 or \
                                                (new_x_max - new_x_min)*(new_y_max - new_y_min) < 10000:
                                            new_new_image = new_image.copy()
                                            draw = ImageDraw.Draw(new_new_image)
                                            draw.rectangle((new_x_min, new_y_min, new_x_max, new_y_max), outline="red", width=2)
                                            new_image_path = f"./lsil/error/{name[:-4]}_{coord_tar}_{i}_{j}.png"
                                            new_new_image.save(new_image_path)
                                            flag = True
                                            break

                                        new_annotations.append({
                                            'name': 'lsil',
                                            'xmin': str(new_x_min),
                                            'ymin': str(new_y_min),
                                            'xmax': str(new_x_max),
                                            'ymax': str(new_y_max)
                                        })

                                    if flag:
                                        new_annotations = []
                                        break

                            if new_annotations:
                                new_annotations = remove_duplicates(new_annotations)

                                new_new_image = new_image.copy()
                                draw = ImageDraw.Draw(new_new_image)
                                new_image_path = f"./lsil/mask/{name[:-4]}_{coord_tar}_{i}_{j}.png"
                                for annotation in new_annotations:
                                    draw.rectangle((eval(annotation['xmin']), eval(annotation['ymin']),
                                                    eval(annotation['xmax']), eval(annotation['ymax'])),
                                                   outline="red", width=2)
                                new_new_image.save(new_image_path)

                                # 保存图像
                                new_image_path = f"./lsil/images/{name[:-4]}_{coord_tar}_{i}_{j}.png"
                                new_image.save(new_image_path)

                                create_voc_xml(new_annotations, f"{name[:-4]}_{coord_tar}_{i}_{j}.png", new_image_path,
                                               f"./lsil/Annotations/{name[:-4]}_{coord_tar}_{i}_{j}.xml", x2 - x1, y2 - y1)

                                print(new_image_path, new_annotations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split model')
    parser.add_argument('--split_path', default=r"D:\Dataset\wsi\lsil", type=str)
    parser.add_argument('--patch_level', type=int, default=0, help='downsample level at which to patch')
    parser.add_argument('--img_size', type=int, default=1280)
    parser.add_argument('--patch_size', type=int, default=640, help='patch_size')
    parser.add_argument('--slide_size', type=int, default=320, help='slide_size')
    args = parser.parse_args()

    if os.path.exists('lsil/images'):
        shutil.rmtree('lsil/images')
    os.makedirs('lsil/images')
    if os.path.exists('lsil/error'):
        shutil.rmtree('lsil/error')
    os.makedirs('lsil/error')
    if os.path.exists('lsil/Annotations'):
        shutil.rmtree('lsil/Annotations')
    os.makedirs('lsil/Annotations')
    if os.path.exists('lsil/mask'):
        shutil.rmtree('lsil/mask')
    os.makedirs('lsil/mask')

    ais = ["14-CG22_16245_01_[7999, 4479, 9279, 5759]_1_0", "16-CG22_16245_06_[12498, 4159, 13778, 5439]_0_1",
           "18-CG22_16245_10_[3598, 639, 4878, 1919]_1_0", "35-CG22_09454_06_[1657, 2239, 2937, 3519]_1_1",
           "49-CG23_19251_01_[959, 3839, 2239, 5119]_1_0", "58-CG23_20006_06_[2879, 1919, 4159, 3199]_1_0",
           "67-CG21_07505-09_[1279, 9599, 2559, 10879]_0_0", "73-CG21_01779_07_[1617, 3199, 2897, 4479]_1_0",
           "73-CG21_01779_07_[1937, 3199, 3217, 4479]_0_0", "73-CG21_01779_07_[2257, 3199, 3537, 4479]_1_0",
           "77-CG20_07068_01_[3519, 2879, 4799, 4159]_1_1", "4-CG23_14499_02_[3280, 6399, 4560, 7679]_0_0",
           "10-CG23_14498_05_[2559, 10879, 3839, 12159]_0_0", "4-CG23_14499_02_[3280, 6399, 4560, 7679]_1_1",
           "5-CG23_14499_03_[2902, 5440, 4182, 6720]_1_1", "7-CG23_14499_12_[254, 9919, 1534, 11199]_0_1",
           "9-CG23_14498_04_[1279, 6719, 2559, 7999]_1_1", "10-CG23_14498_05_[1279, 10879, 2559, 12159]_1_1",
           "10-CG23_14498_05_[2559, 10879, 3839, 12159]_1_1", "14-CG22_16245_01_[5759, 4159, 7039, 5439]_1_1",
           "14-CG22_16245_01_[7999, 4479, 9279, 5759]_1_1", "16-CG22_16245_06_[12498, 4159, 13778, 5439]_1_1",
           "18-CG22_16245_10_[3598, 639, 4878, 1919]_1_1", "4-CG23_14499_02_[3280, 6399, 4560, 7679]_1_1",
           "4-CG23_14499_02_[3600, 4159, 4880, 5439]_1_1"]

    hsil = ["3-CG23_12974_03_[960, 1280, 2240, 2560]_1_0", "14-CG23_12096_02_[1937, 1983, 3217, 3263]_0_0",
            "14-CG23_12096_02_[2257, 1983, 3537, 3263]_1_0", "22-CG23_12350_01_[352, 959, 1632, 2239]_1_1",
            "22-CG23_12350_01_[1952, 4159, 3232, 5439]_0_1", "26-CG23_12706_01_[0, 639, 1280, 1919]_0_0",
            "26-CG23_12706_01_[320, 959, 1600, 2239]_1_1", "26-CG23_12706_01_[639, 3199, 1919, 4479]_0_0",
            "58_[4016, 5322, 5296, 6602]_1_0", "65_[4168, 1280, 5448, 2560]_1_0", "67_[2239, 6399, 3519, 7679]_1_0"]

    lsil = ["1-2023_10411_01_[3520, 7360, 4800, 8640]_1_1", '1-2023_10411_01_[5760, 1600, 7040, 2880]_0_1',
            "1-2023_10411_01_[5760, 5440, 7040, 6720]_0_1", "2-CG23_10410_02_[960, 985, 2240, 2265]_1_1",
            "2-CG23_10410_02_[3840, 7065, 5120, 8345]_0_0", "2-CG23_10410_02_[5120, 7065, 6400, 8345]_0_0",
            "6-CG23_10031_01_[2249, 5136, 3529, 6416]_0_0", "6-CG23_10031_01_[2569, 5456, 3849, 6736]_1_1",
            "6-CG23_10031_01_[2889, 8656, 4169, 9936]_0_1", "27-CG23_11738_02_[7816, 4800, 9096, 6080]_0_1",
            "33_[1280, 3520, 2560, 4800]_0_1", "33_[2880, 1600, 4160, 2880]_1_1", "33_[3200, 1600, 4480, 2880]_0_1",
            "33_[3520, 1600, 4800, 2880]_1_1", "33_[3520, 4480, 4800, 5760]_1_0", "34_[14668, 10303, 15948, 11583]_1_0",
            "34_[15948, 9023, 17228, 10303]_1_0", "35_[2042, 8000, 3322, 9280]_1_1", "36_[1165, 4800, 2445, 6080]_0_1",
            "36_[6925, 320, 8205, 1600]_0_1", "36_[6925, 2240, 8205, 3520]_0_1", "37_[1201, 5120, 2481, 6400]_0_0",
            "39_[12502, 15615, 13782, 16895]_1_0", "39_[12502, 15935, 13782, 17215]_1_1",
            "39_[12502, 16255, 13782, 17535]_1_0", "46_[10852, 13785, 12132, 15065]_1_1",
            "46_[10852, 13145, 12132, 14425]_1_1", '45_[10241, 10240, 11521, 11520]_1_0',
            "10-2023_10411_01_[6720, 4480, 8000, 5760]_1_0"]


    error = []
    for i in os.listdir(os.path.join(args.split_path, 'data')):
        try:
            print(f"-------- process svs {i} --------")
            # if i == '2-CG23_17664_03.svs':
            wsi = openslide.open_slide(os.path.join(args.split_path, 'data', i))
            h5_path = os.path.join(args.split_path, 'patches_level{} 640 320'.format(args.patch_level), i.replace('svs', 'h5'))
            scale, shape = wsi.level_downsamples[args.patch_level], wsi.level_dimensions[args.patch_level]
            image = wsi.read_region((0, 0), args.patch_level, wsi.level_dimensions[args.patch_level]).convert('RGB')
            anno_region, mask_tumor = get_anno_box_datas(os.path.join(args.split_path, 'json', i.replace('svs', 'json')), scale)
            split_image_and_annotations(i, image, h5_path, shape, anno_region, mask_tumor, args.img_size, args.patch_size, args.slide_size)
        except:
            error.append(i)
    print(error)


