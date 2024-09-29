import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import openslide
import math
from wsi_core.wsi_utils import save_hdf5
from wsi_core.util_classes import isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard, Contour_Checking_fn
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000


class WholeSlideImage(object):
    def __init__(self, path, patch_level):
        self.wsi = openslide.open_slide(path)
        
        self.level_downsamples = [(i, i) for i in self.wsi.level_downsamples]
        self.level_dim = self.wsi.level_dimensions
        self.scale, self.shape = self.wsi.level_downsamples[patch_level], self.wsi.level_dimensions[patch_level]

        self.img = self.wsi.read_region((0, 0), patch_level, self.level_dim[patch_level])

        self.contours_tissue = None
        self.contours_tumor = None
        self.hdf5_file = None

    def segmentTissue(self, seg_level=0, sthresh=20, sthresh_up=255, mthresh=7, close=0, use_otsu=False,
                      filter_params={'a_t': 100}, ref_patch_size=512, exclude_ids=[], keep_ids=[]):
        """
            自动识别并提取出图像中代表组织区域的部分
            通过HSV分割组织->中值阈值->二进制阈值
        """
        def _filter_contours(contours, hierarchy, filter_params):
            """
                按面积过滤轮廓
            """
            filtered = []

            # 查找前景轮廓的索引 (parent == -1)
            hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
            all_holes = []

            # 循环 前景轮廓索引
            for cont_idx in hierarchy_1:
                # 真实轮廓
                cont = contours[cont_idx]
                # 包含在此轮廓中的孔的索引 (children of parent contour)
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                # 获取面积 (includes holes)
                a = cv2.contourArea(cont)
                # 计算每个孔的轮廓面积
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                # 前景轮廓区域的实际面积
                a = a - np.array(hole_areas).sum()
                if a == 0:
                    continue
                if tuple((filter_params['a_t'],)) < tuple((a,)):
                    filtered.append(cont_idx)
                    all_holes.append(holes)

            foreground_contours = [contours[cont_idx] for cont_idx in filtered]

            hole_contours = []

            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids]
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                # 按面积划分的最大孔
                unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                filtered_holes = []

                # 过滤 孔
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params['a_h']:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours

        # 将图像从RGB颜色空间转换到HSV颜色空间
        img_hsv = cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2HSV)
        # 应用中值模糊
        img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # Apply median blurring

        # 二值化阈值（得到白色区域）
        if use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        # 形态学闭合
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)  # 执行形态学操作

        # 计算缩放比例和参考补丁尺寸
        scale = self.level_downsamples[seg_level]
        scaled_ref_patch_area = int(ref_patch_size ** 2 / (scale[0] * scale[1]))
        filter_params = filter_params.copy()
        filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
        filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area

        # 查找轮廓
        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        if filter_params:
            # 根据参数过滤轮廓和孔洞
            foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)

        # 匹配原始图像的尺寸
        # 对每个轮廓中的每个点应用缩放因子，调整轮廓的尺寸
        self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
        # 对每个轮廓中的每个孔洞的每个点应用缩放因子，从而调整孔洞的尺寸
        self.holes_tissue = self.scaleHolesDim(hole_contours, scale)

        # 确定最终的轮廓ID
        if len(keep_ids) > 0:
            contour_ids = set(keep_ids) - set(exclude_ids)
        else:
            contour_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)

        # 根据轮廓ID过滤最终的轮廓列表
        self.contours_tissue = [self.contours_tissue[i] for i in contour_ids]
        # 根据轮廓ID过滤最终的孔洞列表
        self.holes_tissue = [self.holes_tissue[i] for i in contour_ids]

    def isInHoles(self, holes, pt, patch_size):
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0] + patch_size / 2, pt[1] + patch_size / 2), False) > 0:
                return 1
        return 0

    def isInContours(self, cont_check_fn, pt, holes=None, patch_size=256):
        if cont_check_fn(pt):
            if holes is not None:
                return not self.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype='int32') for hole in holes] for holes in contours]

    def process_contours(self, save_path_hdf5, svs_id, patch_level=0, patch_size=256, step_size=256, **kwargs):
        """
            处理轮廓
            args:
                save_path: 保存生成的HDF5文件的路径。
                patch_level: (可选) 指定从WSI的哪一层级提取图像块，默认为0。
                patch_size: (可选) 每个图像块的大小，默认为256像素。
                step_size: (可选) 提取图像块时的步长，默认为256像素。
                **kwargs: (可选) 其他关键字参数，允许函数接受额外的可选参数。
        """
        print("Creating patches for: ", svs_id, "...", )
        n_contours = len(self.contours_tissue)
        print("Total number of contours to process: ", n_contours)
        fp_chunk_size = math.ceil(n_contours * 0.05)
        init = True
        for idx, cont in enumerate(self.contours_tissue):
            if (idx + 1) % fp_chunk_size == fp_chunk_size:
                print('Processing contour {}/{}'.format(idx, n_contours))

            asset_dict, attr_dict = self.process_contour(cont, self.holes_tissue[idx], patch_level, save_path_hdf5,
                                                         patch_size, step_size, svs_id, **kwargs)
            if len(asset_dict) > 0:
                # 第一次就覆盖，后续追加
                if init:
                    save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
                    init = False
                else:
                    save_hdf5(save_path_hdf5, asset_dict, mode='a')

    def process_contour(self, cont, contour_holes, patch_level, save_path_hdf5, patch_size=256, step_size=256, svs_id=None,
                        contour_fn='four_pt_easy', use_padding=True, top_left=None, bot_right=None):
        """
            处理单个轮廓，为其创建图像块。
            Args:
                cont: 当前的轮廓。
                contour_holes: 轮廓中的孔洞。
                patch_level: 提取图像块的层级。
                save_path: 保存图像块的路径。
                patch_size: 图像块的尺寸。
                step_size: 提取图像块的步长。
                contour_fn: 用于检查点是否在轮廓内的函数。
                use_padding: 是否在轮廓周围使用填充。
                top_left: 处理区域的左上角坐标。
                bot_right: 处理区域的右下角坐标。
        """
        # 计算轮廓的边界框，如果轮廓为空，则使用整个层级的范围
        if cont is not None:
            start_x, start_y, w, h = cv2.boundingRect(cont)
        else:
            start_x, start_y, w, h = 0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1]

        # 不同层级之间的缩放比例
        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size * patch_downsample[0], patch_size * patch_downsample[1])

        # 调整轮廓处理区域
        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y + h
            stop_x = start_x + w
        else:
            stop_y = min(start_y + h, img_h - ref_patch_size[1] + 1)
            stop_x = min(start_x + w, img_w - ref_patch_size[0] + 1)

        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))

        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)

        if bot_right is not None or top_left is not None:
            w, h = stop_x - start_x, stop_y - start_y
            # 处理区域的宽度或高度小于等于0，则跳过当前轮廓
            if w <= 0 or h <= 0:
                print("Contour is not in specified ROI, skip")
                return {}, {}
            else:
                print("Adjusted Bounding Box:", start_x, start_y, w, h)

        # 用于判断点是否在轮廓内的函数
        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt_easy':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'four_pt_hard':
                cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
            elif contour_fn == 'basic':
                cont_check_fn = isInContourV1(contour=cont)
            else:
                raise NotImplementedError
        else:
            assert isinstance(contour_fn, Contour_Checking_fn)
            cont_check_fn = contour_fn

        # 计算提取图像块时的步长
        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        # 创建坐标点的网格
        x_range = np.arange(start_x, stop_x, step=step_size_x)
        y_range = np.arange(start_y, stop_y, step=step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

        results = []
        for coord in coord_candidates:
            result = self.process_coord_candidate(coord, contour_holes, ref_patch_size[0], cont_check_fn)
            if result is not None:
                results.append(result)
        results = np.array(results)
        
        # num_workers = mp.cpu_count()
        # if num_workers > 4:
        #     num_workers = 4
        # pool = mp.Pool(num_workers)
                
        # # 第0层级坐标
        # iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
        # # 确定一个给定的坐标点是否在指定的轮廓内，同时不在其关联的孔洞内（如果存在孔洞的话）
        # results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
        # pool.close()
        # # 过滤掉None结果，得到实际坐标
        # results = np.array([result for result in results if result is not None])

        print('Extracted {} coordinates'.format(len(results)))
        if len(results) > 0:
            asset_dict = {'coords': results}

            attr = {'patch_size': patch_size,
                    'patch_level': patch_level,
                    'downsample': self.level_downsamples[patch_level],
                    'downsampled_level_dim': tuple(np.array(self.level_dim[patch_level])),
                    'level_dim': self.level_dim[patch_level],
                    'name': svs_id, 'save_path': os.path.split(save_path_hdf5)[0]}

            attr_dict = {'coords': attr}
            return asset_dict, attr_dict

        else:
            return {}, {}

    def process_coord_candidate(self, coord, contour_holes, ref_patch_size, cont_check_fn):
        # 检查 coord 是否在由 cont_check_fn 定义的轮廓内，并且不在 contour_holes 指定的任何孔洞内
        if self.isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
            return coord
        else:
            return None


class Combine_MIL_Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'].values[idx]
        slide_id = slide_id.split(".")[0]

        label = self.slide_data['label'].values[idx]

        type = self.slide_data['type'].values[idx]

        features_path = os.path.join(self.data_dir, "features")

        full_path = os.path.join(features_path, 'pt_files', type, '{}.pt'.format(slide_id))
        features = torch.load(full_path)
        return features, label
