import math
import os
import openslide
import numpy as np
import argparse
import cv2
import json
import h5py
import pandas as pd
import shutil


def get_anno_box_datas(json_filepath, shape, scale=1):
    """
    取得标注数据,返回矩形候选框格式
    """
    with open(json_filepath) as f:
        dicts = json.load(f)
    tumor_polygons = dicts['positive']
    mask_tumor = np.zeros((shape[1], shape[0]))
    anno_regions = []
    for tumor_polygon in tumor_polygons:
        vertices = np.array(tumor_polygon["vertices"]) / scale
        vertices = vertices.astype(np.int32)
        # 多个多边形填充  (code, code, code)
        cv2.fillPoly(mask_tumor, [vertices], (255, 255, 255))
        # 最小矩形框
        anno_regions.append([vertices.min(axis=0)[0], vertices.min(axis=0)[1],
                             vertices.max(axis=0)[0], vertices.max(axis=0)[1]])
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


def split_data(csv_path, slide_length=4, aug_num=2, num=3, error=[], type="train"):
    df = pd.read_csv(csv_path, encoding="utf-8")
    for i, svs_files in enumerate(df["slide_id"].values):
        try:
            h5_path = os.path.join(args.source, 'patches_level{}'.format(args.patch_level),
                                   svs_files.replace("svs", 'h5'))
            json_path = os.path.join(args.source, "json", svs_files.replace("svs", 'json'))
            svs_path = os.path.join(args.source, "data", svs_files)
            mask_path = os.path.join(args.source, "mask", svs_files[:-4])

            if os.path.exists(os.path.join(mask_path, 'positive')):
                shutil.rmtree(os.path.join(mask_path, 'positive'))
            os.makedirs(os.path.join(mask_path, 'positive'))
            
            if os.path.exists(os.path.join(mask_path, 'negative')):
                shutil.rmtree(os.path.join(mask_path, 'negative'))
            os.makedirs(os.path.join(mask_path, 'negative'))

            print(f"type {type} [{i + 1}/{df.shape[0]}] process {svs_files}")

            wsi = openslide.open_slide(svs_path)
            scale, shape = wsi.level_downsamples[args.patch_level], wsi.level_dimensions[args.patch_level]
            image = np.array(wsi.read_region((0, 0), args.patch_level, wsi.level_dimensions[args.patch_level]))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            anno_regions_positive, mask_tumor = get_anno_box_datas(json_path, shape, scale)
            anno_regions_positive_copy = anno_regions_positive.copy()
            image_copy = image.copy()
            for j, anno in enumerate(anno_regions_positive):
                if 224 - anno[3] + anno[1] > 0:
                    floor1 = math.floor((224 - anno[3] + anno[1]) / 2)
                    ceil1 = math.ceil((224 - anno[3] + anno[1]) / 2)
                    if anno[1] - floor1 < 0:
                        floor1 = 0
                    elif anno[3] + ceil1 > shape[1]:
                        floor1 = shape[1] - anno[3]
                else:
                    floor1 = 0
                    ceil1 = 0
                if 224 - anno[2] + anno[0] > 0:
                    floor2 = math.floor((224 - anno[2] + anno[0]) / 2)
                    ceil2 = math.ceil((224 - anno[2] + anno[0]) / 2)
                    if anno[0] - floor2 < 0:
                        floor2 = 0
                    elif anno[2] + ceil2 > shape[0]:
                        floor1 = shape[0] - anno[2]
                else:
                    floor2 = 0
                    ceil2 = 0

                temp_anno = [anno[0] - floor2, anno[1] - floor1, anno[2] + ceil2, anno[3] + ceil1]
                anno_regions_positive_copy.append(temp_anno)
                cv2.rectangle(image_copy, temp_anno[:2], temp_anno[2:], (0, 0, 0), 20, 2)

                if type == 'train':
                    img_zero = image[anno[1]:anno[3], anno[0]:anno[2], :]
                    img_zero = letterbox(img_zero, new_shape=args.img_size)
                    cv2.imwrite(f'{os.path.join(mask_path, "positive")}/positive_{j}_zero.jpg', img_zero)

                img = image[temp_anno[1]:temp_anno[3], temp_anno[0]:temp_anno[2], :]
                cv2.imwrite(f'{os.path.join(mask_path, "positive")}/positive_{j}.jpg', img)

                # mask_img = mask_tumor[anno[1]:anno[3], anno[0]:anno[2]]
                # cv2.imwrite(f'{os.path.join(mask_path, "positive")}/{j}_mask.jpg', mask_img)

            if type == 'train':
                aug_annotation_patches(os.path.join(mask_path, "positive"), aug_num)

            anno_regions_negative, negative_coords = [], []
            negative_coords_temp = []
            with h5py.File(h5_path, "a") as f:
                coords = np.array(f['coords'])
                for coord in coords:
                    coord = np.array(coord / scale).astype(np.int32)
                    for j in range(slide_length):
                        for k in range(slide_length):
                            coord_tar = np.array([coord[0] + j * args.patch_size, coord[1] + k * args.patch_size]).astype(np.int16)
                            coord_tar = [coord_tar[0], coord_tar[1], coord_tar[0] + args.patch_size,
                                         coord_tar[1] + args.patch_size]
                            if coord_tar[2] < shape[0] and coord_tar[3] < shape[1]:
                                if do_rectangles_intersect(coord_tar, anno_regions_positive_copy):
                                    continue
                                else:
                                    negative_coords_temp.append(coord_tar)

                np.random.shuffle(negative_coords_temp)
                for negative_coords in negative_coords_temp[:len(os.listdir(os.path.join(mask_path, "positive")) * num)]:
                    try:
                        img = image[negative_coords[1]:negative_coords[3], negative_coords[0]:negative_coords[2], :]
                        cv2.imwrite(f'{os.path.join(mask_path, "negative")}/negative_{negative_coords}.jpg', img)
                    except:
                        continue
                    cv2.rectangle(image_copy, negative_coords[:2], negative_coords[2:], (0, 255, 0), 20, 2)

                cv2.imwrite(f'{mask_path}/negative.jpg', image_copy)

                if "positive" in f:
                    del f["positive"]
                if "negative" in f:
                    del f["negative"]
                f.create_dataset("positive", data=anno_regions_positive)
                f.create_dataset("negative", data=anno_regions_negative)
        except Exception as e:
            error.append([type, svs_files, e])
    return error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='seg and patch')
    parser.add_argument('--source', type=str, default=r"/home/bavon/datasets/wsi/ais",
                        help='path to folder containing raw wsi image files')
    parser.add_argument('--img_size', type=int, default=224, help='img_size')
    parser.add_argument('--patch_level', type=int, default=1, help='downsample level at which to patch')
    parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
    parser.add_argument('--slide_size', type=int, default=64, help='patch_size')
    parser.add_argument('--save_dir', type=str, default=r"/home/bavon/datasets/wsi/ais",
                        help='directory to save processed data')
    args = parser.parse_args()

    # build_data_csv(args.source)

    train_cvc = os.path.join(args.source, "train.csv")
    val_cvc = os.path.join(args.source, "valid.csv")

    slide_length = args.patch_size // args.slide_size

    error = []
    error = split_data(train_cvc, slide_length=slide_length, aug_num=2, num=3, error=error, type="train")
    error = split_data(val_cvc, slide_length=slide_length, num=1, error=error, type='val')

    print("process success!!!")
    print('process fail file: ', error)
