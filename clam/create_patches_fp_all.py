import math
from tqdm import tqdm
import openslide
import numpy as np
import argparse
import cv2
import json
import h5py
import pandas as pd
import shutil
import os
import pickle


def find_temp(min_matrix_1, min_matrix_2, patch_size, temp=2):
    # 计算当前的gap
    gap = (min_matrix_1 - min_matrix_2) / temp

    # 如果gap小于或等于patch_size，返回当前的temp
    if gap <= patch_size:
        return temp, gap
    else:
        # 否则，递归调用find_temp，temp加1
        return find_temp(min_matrix_1, min_matrix_2, patch_size, temp + 1)


def get_anno_box_datas(json_filepath, shape, scale, patch_size):
    """
    取得标注数据,返回矩形候选框格式
    """
    with open(json_filepath) as f:
        dicts = json.load(f)
    tumor_polygons = dicts['positive']
    mask_tumor = np.zeros((shape[1], shape[0]))
    anno_regions = []
    for tumor_polygon in tqdm(tumor_polygons, desc='process positive', total=len(tumor_polygons)):
        vertices = np.array(tumor_polygon["vertices"]) / scale
        vertices = vertices.astype(np.int32)
        # 多个多边形填充  (code, code, code)
        cv2.fillPoly(mask_tumor, [vertices], (255, 255, 255))
        # 最小矩形框
        min_matrix = [vertices.min(axis=0)[0], vertices.min(axis=0)[1],
                      vertices.max(axis=0)[0], vertices.max(axis=0)[1]]
        # # x > 128 > y
        # if (min_matrix[2] - min_matrix[0]) > patch_size > (min_matrix[3] - min_matrix[1]):
        #     temp, gap = find_temp(min_matrix[2], min_matrix[0], patch_size)
        #     for i in range(1, temp + 1):
        #         min_matrix_temp = [int(min_matrix[0] + gap * (i - 1)), min_matrix[1],
        #                            int(min_matrix[0] + gap * i), min_matrix[3]]
        #         anno_regions.append(min_matrix_temp)
        # # x < 224 < y
        # elif (min_matrix[2] - min_matrix[0]) < patch_size < (min_matrix[3] - min_matrix[1]):
        #     temp, gap = find_temp(min_matrix[3], min_matrix[1], patch_size)
        #     for i in range(1, temp + 1):
        #         min_matrix_temp = [min_matrix[0], int(min_matrix[1] + gap * (i - 1)),
        #                            min_matrix[2], int(min_matrix[1] + gap * i)]
        #         anno_regions.append(min_matrix_temp)
        # # x > 224 and y > 224
        # elif (min_matrix[2] - min_matrix[0]) > patch_size and (min_matrix[3] - min_matrix[1]) > patch_size:
        #     temp_x, gap_x = find_temp(min_matrix[2], min_matrix[0], patch_size)
        #     temp_y, gap_y = find_temp(min_matrix[3], min_matrix[1], patch_size)
        #     for i in range(1, temp_x + 1):
        #         for j in range(1, temp_y + 1):
        #             min_matrix_temp = [int(min_matrix[0] + gap_x * (i - 1)), int(min_matrix[1] + gap_y * (j - 1)),
        #                                int(min_matrix[0] + gap_x * i), int(min_matrix[1] + gap_y * j)]
        #             anno_regions.append(min_matrix_temp)
        # else:
        anno_regions.append(min_matrix)

    return anno_regions, mask_tumor


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0), auto=False, scaleFill=False, scaleup=True, stride=32):
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


def do_rectangles_intersect(rect1, rect2):
    # 解包矩形坐标
    x1, y1, x2, y2 = rect1
    for rect in rect2:
        x3, y3, x4, y4 = rect
        # 检查是否有交集
        if not (x2 <= x3 or x1 >= x4 or y2 <= y3 or y1 >= y4):
            return True
    else:
        return False


def aug_annotation_patches(file_path, number):
    import Augmentor
    p = Augmentor.Pipeline(file_path, output_directory=file_path)

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


def build_data_csv(file_path, split_rate=0.8):
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


def find_ratio_of_value_numpy(array_2d, value=255):
    # 将列表转换为NumPy数组
    np_array = np.array(array_2d)

    # 计算值为value的元素数量
    count_value = np_array[np_array == value].size
    # 计算比率
    ratio = count_value / np_array.size
    return ratio


def split_data(csv_path, patch_size=64, slide_size=16, patch_level=0, img_size=224, aug_num=2, num=3,
               error=[], type="train", name='ais'):
    slide_length = patch_size // slide_size
    df = pd.read_csv(csv_path, encoding="utf-8")
    for i, svs_files in enumerate(df["slide_id"].values):
        # if svs_files in ['4-CG23_14499_02.svs']:
        try:
            h5_path = os.path.join(args.source, 'patches_level{}'.format(patch_level), svs_files.replace("svs", 'h5'))
            json_path = os.path.join(args.source, "json", svs_files.replace("svs", 'json'))
            svs_path = os.path.join(args.source, "data", svs_files)
            mask_path = os.path.join(args.source, "mask", svs_files[:-4])

            if os.path.exists(os.path.join(mask_path, 'positive')):
                shutil.rmtree(os.path.join(mask_path, 'positive'))
            os.makedirs(os.path.join(mask_path, 'positive'))

            if os.path.exists(os.path.join(mask_path, 'positive_mask')):
                shutil.rmtree(os.path.join(mask_path, 'positive_mask'))
            os.makedirs(os.path.join(mask_path, 'positive_mask'))

            if os.path.exists(os.path.join(mask_path, 'negative')):
                shutil.rmtree(os.path.join(mask_path, 'negative'))
            os.makedirs(os.path.join(mask_path, 'negative'))

            if not os.path.exists(os.path.join(args.source, 'org')):
                os.makedirs(os.path.join(args.source, 'org'))
                
            print(f"\ntype {type} [{i + 1}/{df.shape[0]}] process {svs_files}")

            wsi = openslide.open_slide(svs_path)
            scale, shape = wsi.level_downsamples[args.patch_level], wsi.level_dimensions[args.patch_level]
            image = np.array(wsi.read_region((0, 0), args.patch_level, wsi.level_dimensions[args.patch_level]))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            anno_region, mask_tumor = get_anno_box_datas(json_path, shape, scale, patch_size)
            image_copy = image.copy()

            anno_region_positive, error_anno_regions_positive, anno_region_negative = [], [], []
            anno_region_positive_img, anno_region_negative_img = [], []
            for j, anno in enumerate(tqdm(anno_region, desc='save svs', total=len(anno_region))):
                # 补全，使其图片尺寸达到224，224
                if patch_size - anno[3] + anno[1] > 0:
                    floor1 = math.floor((patch_size - anno[3] + anno[1]) / 2)
                    ceil1 = math.ceil((patch_size - anno[3] + anno[1]) / 2)
                    if anno[1] - floor1 < 0:
                        floor1 = 0
                    elif anno[3] + ceil1 > shape[1]:
                        floor1 = shape[1] - anno[3]
                else:
                    floor1 = 0
                    ceil1 = 0
                if patch_size - anno[2] + anno[0] > 0:
                    floor2 = math.floor((patch_size - anno[2] + anno[0]) / 2)
                    ceil2 = math.ceil((patch_size - anno[2] + anno[0]) / 2)
                    if anno[0] - floor2 < 0:
                        floor2 = 0
                    elif anno[2] + ceil2 > shape[0]:
                        floor1 = shape[0] - anno[2]
                else:
                    floor2 = 0
                    ceil2 = 0

                # 新坐标
                temp_anno = [anno[0] - floor2, anno[1] - floor1, anno[2] + ceil2, anno[3] + ceil1]

                temp_mask = mask_tumor[temp_anno[1]: temp_anno[3], temp_anno[0]:temp_anno[2]]
                img = image[temp_anno[1]:temp_anno[3], temp_anno[0]:temp_anno[2], :]

                if name != 'lsil':
                    anno_region_positive_img.append(img)
                    anno_region_positive.append(temp_anno)
                    cv2.rectangle(image_copy, temp_anno[:2], temp_anno[2:], (0, 0, 0), 20, 2)
                    cv2.imwrite(f'{os.path.join(mask_path, "positive")}/positive_{j}.png', img)
                    cv2.imwrite(f'{os.path.join(mask_path, "positive_mask")}/mask_{j}.png', temp_mask)
                else:
                    if find_ratio_of_value_numpy(temp_mask) > 0.5:
                        anno_region_positive.append(temp_anno)
                        cv2.rectangle(image_copy, temp_anno[:2], temp_anno[2:], (0, 0, 0), 20, 2)
                        cv2.imwrite(f'{os.path.join(mask_path, "positive")}/positive_{j}.png', img)
                        cv2.imwrite(f'{os.path.join(mask_path, "positive_mask")}/mask_{j}.png', temp_mask)
                    else:
                        error_anno_regions_positive.append(temp_anno)
                        cv2.rectangle(image_copy, temp_anno[:2], temp_anno[2:], (0, 0, 255), 20, 2)

                if type == 'train':
                    # 原始标注尺寸
                    anno_region_positive_img.append(anno)
                    img_zero = image[anno[1]:anno[3], anno[0]:anno[2], :]
                    img_zero = letterbox(img_zero, new_shape=img_size)
                    cv2.imwrite(f'{os.path.join(mask_path, "positive")}/positive_{j}_zero.png', img_zero)

            if type == 'train':
                # 数据增强
                aug_annotation_patches(os.path.join(mask_path, "positive"), aug_num)

            negative_coords_other, negative_coords_blank = [], []
            with h5py.File(h5_path, "a") as f:
                coords = np.array(f['coords'])
                for coord in coords:
                    coord = np.array(coord / scale).astype(int)
                    for j in range(slide_length):
                        for k in range(slide_length):
                            coord_tar = np.array([coord[0] + j * slide_size, coord[1] + k * slide_size]).astype(
                                np.int16)
                            coord_tar = [coord_tar[0], coord_tar[1],
                                         coord_tar[0] + patch_size, coord_tar[1] + patch_size]
                            if coord_tar[2] < shape[0] and coord_tar[3] < shape[1]:
                                if name != "lsil":
                                    if do_rectangles_intersect(coord_tar, anno_region_positive):
                                        continue
                                    else:
                                        if coord_tar[0] == 0:
                                            negative_coords_blank.append(coord_tar)
                                        else:
                                            negative_coords_other.append(coord_tar)
                                # else:
                                #     if do_rectangles_intersect(coord_tar, anno_region_positive) or \
                                #             do_rectangles_intersect(coord_tar, error_anno_regions_positive):
                                #         continue
                                #     else:
                                #         if coord_tar[0] == 0:
                                #             negative_coords_blank.append(coord_tar)
                                #         else:
                                #             negative_coords_other.append(coord_tar)

            np.random.shuffle(negative_coords_blank)
            np.random.shuffle(negative_coords_other)

            #  计算 negative 需要的数量
            len_num = len(os.listdir(os.path.join(mask_path, "positive")))
            blank_num = len_num * (num - 1) if len_num else 10
            other_num = len_num * (num + 1) if len_num else 20
            anno_region_negative = negative_coords_blank[:blank_num] + negative_coords_other[:other_num]
            for negative_coords in anno_region_negative:
                img = image[negative_coords[1]:negative_coords[3], negative_coords[0]:negative_coords[2], :]
                anno_region_negative_img.append(img)
                cv2.imwrite(f'{os.path.join(mask_path, "negative")}/negative_{negative_coords}.png', img)
                cv2.rectangle(image_copy, negative_coords[:2], negative_coords[2:], (0, 255, 0), 20, 2)
            
            cv2.imwrite(os.path.join(args.source, 'org', svs_files.replace("svs", 'png')), image_copy)
            
            anno_region = {'positive': anno_region_positive, 'negative': anno_region_negative}
            with open(f'{mask_path}/coords.txt', 'w') as file:
                file.write(str(anno_region))
                
            images_data = {
                'positive': anno_region_positive_img,
                'negative': anno_region_negative_img
            }
            with open(f'{mask_path}/coords.pkl', 'wb') as f:
                pickle.dump(images_data, f)
        
        except Exception as e:
            error.append([type, svs_files, e])
    return error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='seg and patch')
    parser.add_argument('--source', type=str, default=r"/home/qdata/datasets/wsi/ais",
                        help='path to folder containing raw wsi image files')
    parser.add_argument('--img_size', type=int, default=224, help='img_size')
    parser.add_argument('--patch_level', type=int, default=1, help='downsample level at which to patch')
    parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
    parser.add_argument('--slide_size', type=int, default=64, help='slide_size')
    parser.add_argument('--save_dir', type=str, default=r"/home/qdata/datasets/wsi/ais",
                        help='directory to save processed data')
    args = parser.parse_args()

    # build_data_csv(args.source)

    train_cvc = os.path.join(args.source, "train.csv")
    val_cvc = os.path.join(args.source, "valid.csv")

    error = []
    error = split_data(train_cvc, patch_size=args.patch_size, slide_size=args.slide_size,
                        patch_level=args.patch_level, img_size=args.img_size, aug_num=2, num=10, error=error,
                       type="train", name='ais')
    error = split_data(val_cvc, patch_size=args.patch_size, slide_size=args.slide_size,
                       patch_level=args.patch_level, img_size=args.img_size, num=3, error=error, type='val', name='ais')

    print("process success!!!")
    print('process fail file: ', error)
