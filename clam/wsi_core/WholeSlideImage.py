import os
from io import BytesIO
from xml.dom import minidom
import multiprocessing as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
import math
from wsi_core.wsi_utils import savePatchIter_bag_hdf5, initialize_hdf5_bag, save_hdf5, \
    screen_coords, isBlackPatch, isWhitePatch, to_percentiles
from wsi_core.util_classes import isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard, \
    Contour_Checking_fn
from utils.file_utils import load_pkl, save_pkl
from PIL import Image, ImageDraw, ImageFont
from segmodel.segone import piplineone


Image.MAX_IMAGE_PIXELS = 933120000


class WholeSlideImage(object):
    def __init__(self, path):
        """
        Args:
            path (str): fullpath to WSI file
        """
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.wsi = openslide.open_slide(path)
        self.level_downsamples = self._assertLevelDownsamples()
        self.level_dim = self.wsi.level_dimensions

        self.contours_tissue = None
        self.contours_tumor = None
        self.hdf5_file = None

    def getOpenSlide(self):
        return self.wsi

    def initXML(self, xml_path):
        """
            初始化肿瘤轮廓，从 XML 文件中读取数据。
        """

        def _createContour(coord_list):
            """
                将 XML 中的坐标列表转换成 numpy 数组。
                Args:
                    coord_list: 包含坐标点的 minidom 元素列表。
                Returns:
                    numpy.array: 包含整数坐标点的 numpy 数组。
            """
            return np.array([[[int(float(coord.attributes['X'].value)),
                               int(float(coord.attributes['Y'].value))]] for coord in coord_list], dtype='int32')

        # 解析 XML 文件，创建一个文档对象
        xmldoc = minidom.parse(xml_path)
        # 从文档中提取所有的 'Annotation' 元素，并获取每个注释中的 'Coordinate' 子元素列表
        annotations = [anno.getElementsByTagName('Coordinate') for anno in xmldoc.getElementsByTagName('Annotation')]
        # 对于每个 'Coordinate' 列表，使用 _createContour 函数转换成 numpy 数组
        self.contours_tumor = [_createContour(coord_list) for coord_list in annotations]
        # 按照轮廓面积对肿瘤轮廓列表进行降序排序
        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def initTxt(self, annot_path):
        """
            从文本文件中初始化肿瘤轮廓。
        """

        def _create_contours_from_dict(annot):
            """
                根据注释字典创建轮廓列表。
                Args:
                    annot: 包含注释数据的字典。
                Returns:
                    list: 包含多个轮廓的列表，每个轮廓是一个 numpy 数组。
            """
            all_cnts = []
            for idx, annot_group in enumerate(annot):
                contour_group = annot_group['coordinates']
                # 多边形
                if annot_group['type'] == 'Polygon':
                    for idx, contour in enumerate(contour_group):
                        contour = np.array(contour).astype(np.int32).reshape(-1, 1, 2)
                        all_cnts.append(contour)
                else:
                    for idx, sgmt_group in enumerate(contour_group):
                        contour = []
                        for sgmt in sgmt_group:
                            contour.extend(sgmt)
                        contour = np.array(contour).astype(np.int32).reshape(-1, 1, 2)
                        all_cnts.append(contour)

            return all_cnts

        with open(annot_path, "r") as f:
            annot = f.read()
            annot = eval(annot)

        # 创建轮廓列表
        self.contours_tumor = _create_contours_from_dict(annot)
        # 按照轮廓面积对肿瘤轮廓列表进行降序排序
        self.contours_tumor = sorted(self.contours_tumor, key=cv2.contourArea, reverse=True)

    def initSegmentation(self, mask_file):
        # 从pickle文件中初始化分割结果
        asset_dict = load_pkl(mask_file)
        # 获取组织和肿瘤的轮廓信息
        self.holes_tissue = asset_dict['holes']
        self.contours_tissue = asset_dict['tissue']

    def saveSegmentation(self, mask_file):
        # 使用pickle保存分割结果
        asset_dict = {'holes': self.holes_tissue, 'tissue': self.contours_tissue}
        save_pkl(mask_file, asset_dict)

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

        img = np.array(self.wsi.read_region((0, 0), seg_level, self.level_dim[seg_level]))
        # 将图像从RGB颜色空间转换到HSV颜色空间
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
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

    def segmentTissue_new_model(self, seg_level=0, sthresh=20, sthresh_up=255, mthresh=7, close=0, use_otsu=False,
                                filter_params={'a_t': 100}, ref_patch_size=512, exclude_ids=[], keep_ids=[],
                                model=None, device='cpu'):
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

        img = np.array(self.wsi.read_region((0, 0), seg_level, self.level_dim[seg_level]))
        # 将 'RGB' 转换为 'BGR'
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        self.img_otsu = piplineone(model, image, device)

        # 计算缩放比例和参考补丁尺寸
        scale = self.level_downsamples[seg_level]
        scaled_ref_patch_area = int(ref_patch_size ** 2 / (scale[0] * scale[1]))
        filter_params = filter_params.copy()
        filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
        filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area

        # 查找轮廓
        contours, hierarchy = cv2.findContours(self.img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
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

    def visWSI(self, vis_level=0, color=(0, 255, 0), hole_color=(0, 0, 255), annot_color=(255, 0, 0),
               line_thickness=250, max_size=None, top_left=None, bot_right=None, custom_downsample=1,
               view_slide_only=False, number_contours=False, seg_display=True, annot_display=True):
        """
        可视化WSI的特定层级，并可选择性地显示轮廓和注释
            Args:
            vis_level (int): 要可视化的WSI层级，默认为0。
            color (tuple): 组织轮廓的颜色，默认为(0, 255, 0)，即绿色。
            hole_color (tuple): 孔洞轮廓的颜色，默认为(0, 0, 255)，即红色。
            annot_color (tuple): 注释轮廓的颜色，默认为(255, 0, 0)，即蓝色。
            line_thickness (int): 轮廓线条的粗细，默认为250。
            max_size (int): 最大画布尺寸，超过此尺寸将进行缩放，默认为None。
            top_left (tuple): 可视化区域左上角的坐标，默认为None。
            bot_right (tuple): 可视化区域右下角的坐标，默认为None。
            custom_downsample (int): 自定义下采样因子，默认为1。
            view_slide_only (bool): 是否仅查看幻灯片，默认为False。
            number_contours (bool): 是否为每个轮廓添加编号，默认为False。
            seg_display (bool): 是否显示组织分割轮廓，默认为True。
            annot_display (bool): 是否显示注释轮廓，默认为True。
        """
        # 获取指定层级的下采样比例
        downsample = self.level_downsamples[vis_level]
        # 计算从层级0到可视化层级的缩放比例
        scale = [1 / downsample[0], 1 / downsample[1]]

        # 根据top_left和bot_right确定可视化区域的大小
        if top_left is not None and bot_right is not None:
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
        else:
            top_left = (0, 0)
            region_size = self.level_dim[vis_level]

        # 读取指定层级和区域的图像
        img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))

        # 显示轮廓和注释，则进行以下处理
        if not view_slide_only:
            # 计算绘制轮廓时的偏移量
            offset = tuple(-(np.array(top_left) * scale).astype(int))
            # 根据缩放比例调整线条粗细
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            # 如果存在组织轮廓并且需要显示，则绘制组织轮廓
            if self.contours_tissue is not None and seg_display:
                if not number_contours:
                    cv2.drawContours(img, self.scaleContourDim(self.contours_tissue, scale),
                                     -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)

                else:
                    # 为每个轮廓添加编号
                    for idx, cont in enumerate(self.contours_tissue):
                        contour = np.array(self.scaleContourDim(cont, scale))
                        M = cv2.moments(contour)
                        cX = int(M["m10"] / (M["m00"] + 1e-9))
                        cY = int(M["m01"] / (M["m00"] + 1e-9))
                        # 绘制轮廓并将文本放在中心旁边
                        cv2.drawContours(img, [contour], -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                        cv2.putText(img, "{}".format(idx), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)

                # 绘制孔洞轮廓
                for holes in self.holes_tissue:
                    cv2.drawContours(img, self.scaleContourDim(holes, scale), -1, hole_color, line_thickness,
                                     lineType=cv2.LINE_8)

            # 显示 绘制注释轮廓
            if self.contours_tumor is not None and annot_display:
                cv2.drawContours(img, self.scaleContourDim(self.contours_tumor, scale), -1, annot_color, line_thickness,
                                 lineType=cv2.LINE_8, offset=offset)

        img = Image.fromarray(img)
        w, h = img.size

        # 根据下采样 对图像进行缩放
        if custom_downsample > 1:
            img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

        # 图像尺寸超过max_size，进行缩放
        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size / w if w > h else max_size / h
            img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

        return img

    def createPatches_bag_hdf5(self, save_path, patch_level=0, patch_size=256, step_size=256, save_coord=True,
                               **kwargs):
        """
            为WSI创建一个包含图像块的HDF5文件。
            Args:
                save_path (str): 保存HDF5文件的路径。
                patch_level (int): 从哪个层级获取图像块，默认为0。
                patch_size (int): 图像块的尺寸，默认为256像素。
                step_size (int): 图像块提取的步长，默认为256像素。
                save_coord (bool): 是否保存图像块的坐标，默认为True。
                **kwargs: 其他关键字参数，可用于传递其他选项。
        """
        contours = self.contours_tissue

        print("Creating patches for: ", self.name, "...", )
        for idx, cont in enumerate(contours):
            # 每个轮廓 生成图像块
            patch_gen = self._getPatchGenerator(cont, idx, patch_level, save_path, patch_size, step_size, **kwargs)

            if self.hdf5_file is None:
                try:
                    first_patch = next(patch_gen)
                except StopIteration:
                    continue

                file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
                self.hdf5_file = file_path

            # 遍历生成器，保存每个图像块到HDF5文件
            for patch in patch_gen:
                savePatchIter_bag_hdf5(patch)

        return self.hdf5_file

    def _getPatchGenerator(self, cont, cont_idx, patch_level, save_path, patch_size=256, step_size=256,
                           custom_downsample=1,
                           white_black=True, white_thresh=15, black_thresh=50, contour_fn='four_pt', use_padding=True):
        """
            生成一个图像块生成器，用于获取指定轮廓内的图像块。
            Args:
                cont (numpy.array): 轮廓坐标。
                cont_idx (int): 轮廓的索引。
                patch_level (int): 提取图像块的层级。
                save_path (str): 保存图像块的路径。
                patch_size (int): 图像块的尺寸，默认为256像素。
                step_size (int): 提取图像块时的步长，默认为256像素。
                custom_downsample (int): 自定义下采样因子，默认为1。
                white_black (bool): 是否过滤白色或黑色块，默认为True。
                white_thresh (int): 白色块的阈值，默认为15。
                black_thresh (int): 黑色块的阈值，默认为50。
                contour_fn (str or Contour_Checking_fn): 用于检查点是否在轮廓内的函数，默认为'four_pt'。
                use_padding (bool): 是否使用填充，默认为True，表示在轮廓周围添加额外的图像块。
            """
        # 计算轮廓的边界框，如果轮廓为空，则使用整个层级的范围
        if cont is not None:
            start_x, start_y, w, h = cv2.boundingRect(cont)
        else:
            start_x, start_y, w, h = 0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1]

        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))

        if custom_downsample > 1:
            assert custom_downsample == 2
            target_patch_size = patch_size
            patch_size = target_patch_size * 2
            step_size = step_size * 2
            print("Custom Downsample: {}, Patching at {} x {}, But Final Patch Size is {} x {}".
                  format(custom_downsample, patch_size, patch_size, target_patch_size, target_patch_size))

        # 下采样
        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size * patch_downsample[0], patch_size * patch_downsample[1])

        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
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

        # 获取到结束位置
        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y + h
            stop_x = start_x + w
        else:
            stop_y = min(start_y + h, img_h - ref_patch_size[1])
            stop_x = min(start_x + w, img_w - ref_patch_size[0])

        count = 0
        # 遍历每个可能的图像块位置
        for y in range(start_y, stop_y, step_size_y):
            for x in range(start_x, stop_x, step_size_x):
                # 不在轮廓及其相关孔内的点
                if not self.isInContours(cont_check_fn, (x, y), self.holes_tissue[cont_idx], ref_patch_size[0]):
                    continue

                count += 1
                # 获取图像并 resize 到目标尺寸
                patch_PIL = self.wsi.read_region((x, y), patch_level, (patch_size, patch_size)).convert('RGB')
                if custom_downsample > 1:
                    patch_PIL = patch_PIL.resize((target_patch_size, target_patch_size))

                if white_black:
                    if isBlackPatch(np.array(patch_PIL), rgbThresh=black_thresh) or isWhitePatch(np.array(patch_PIL),
                                                                                                 satThresh=white_thresh):
                        continue

                patch_info = {'x': x // (patch_downsample[0] * custom_downsample),
                              'y': y // (patch_downsample[1] * custom_downsample),
                              'cont_idx': cont_idx, 'patch_level': patch_level,
                              'downsample': self.level_downsamples[patch_level],
                              'downsampled_level_dim': tuple(
                                  np.array(self.level_dim[patch_level]) // custom_downsample),
                              'level_dim': self.level_dim[patch_level],
                              'patch_PIL': patch_PIL, 'name': self.name, 'save_path': save_path}

                yield patch_info

        print("patches extracted: {}".format(count))

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0] + patch_size / 2, pt[1] + patch_size / 2), False) > 0:
                return 1

        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
        if cont_check_fn(pt):
            if holes is not None:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype='int32') for hole in holes] for holes in contours]

    def _assertLevelDownsamples(self):
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]

        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (
                downsample, downsample) else level_downsamples.append((downsample, downsample))

        return level_downsamples

    def process_contours(self, save_path, patch_level=0, patch_size=256, step_size=256, **kwargs):
        """
            处理轮廓
            args:
                save_path: 保存生成的HDF5文件的路径。
                patch_level: (可选) 指定从WSI的哪一层级提取图像块，默认为0。
                patch_size: (可选) 每个图像块的大小，默认为256像素。
                step_size: (可选) 提取图像块时的步长，默认为256像素。
                **kwargs: (可选) 其他关键字参数，允许函数接受额外的可选参数。
        """
        save_path_hdf5 = os.path.join(save_path, str(self.name) + '.h5')
        print("Creating patches for: ", self.name, "...", )
        n_contours = len(self.contours_tissue)
        print("Total number of contours to process: ", n_contours)
        fp_chunk_size = math.ceil(n_contours * 0.05)
        init = True
        for idx, cont in enumerate(self.contours_tissue):
            if (idx + 1) % fp_chunk_size == fp_chunk_size:
                print('Processing contour {}/{}'.format(idx, n_contours))

            asset_dict, attr_dict = self.process_contour(cont, self.holes_tissue[idx], patch_level, save_path,
                                                         patch_size, step_size, **kwargs)
            if len(asset_dict) > 0:
                # 第一次就覆盖，后续追加
                if init:
                    save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
                    init = False
                else:
                    save_hdf5(save_path_hdf5, asset_dict, mode='a')

    def process_contour(self, cont, contour_holes, patch_level, save_path, patch_size=256, step_size=256,
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

        # print("Bounding Box:", start_x, start_y, w, h)
        # print("Contour Area:", cv2.contourArea(cont))

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

        num_workers = mp.cpu_count()
        if num_workers > 4:
            num_workers = 4
        pool = mp.Pool(num_workers)

        # 第0层级坐标
        iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
        # 确定一个给定的坐标点是否在指定的轮廓内，同时不在其关联的孔洞内（如果存在孔洞的话）
        results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
        pool.close()
        # 过滤掉None结果，得到实际坐标
        results = np.array([result for result in results if result is not None])

        # print('Extracted {} coordinates'.format(len(results)))
        if len(results) > 0:
            asset_dict = {'coords': results}

            attr = {'patch_size': patch_size,
                    'patch_level': patch_level,
                    'downsample': self.level_downsamples[patch_level],
                    'downsampled_level_dim': tuple(np.array(self.level_dim[patch_level])),
                    'level_dim': self.level_dim[patch_level],
                    'name': self.name, 'save_path': save_path}

            attr_dict = {'coords': attr}
            return asset_dict, attr_dict

        else:
            return {}, {}

    @staticmethod
    def process_coord_candidate(coord, contour_holes, ref_patch_size, cont_check_fn):
        # 检查 coord 是否在由 cont_check_fn 定义的轮廓内，并且不在 contour_holes 指定的任何孔洞内
        if WholeSlideImage.isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
            return coord
        else:
            return None

    def visHeatmap(self, scores, coords, vis_level=-1, top_left=None, bot_right=None, patch_size=(256, 256),
                   blank_canvas=False, alpha=0.4, blur=False, overlap=0.0, segment=True, use_holes=True,
                   convert_to_percentiles=False, binarize=False, thresh=0.5,
                   max_size=None, custom_downsample=1, cmap='coolwarm', k=15):
        """
        Args:
            scores (numpy array of float): Attention scores 
            coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
            vis_level (int): WSI pyramid level to visualize
            patch_size (tuple of int): Patch dimensions (relative to lvl 0)
            blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
            alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
            blur (bool): apply gaussian blurring
            overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
            segment (bool): whether to use tissue segmentation contour (must have already called self.segmentTissue such that 
                            self.contours_tissue and self.holes_tissue are not None
            use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
            convert_to_percentiles (bool): whether to convert attention scores to percentiles
            binarize (bool): only display patches > threshold
            threshold (float): binarization threshold
            max_size (int): Maximum canvas size (clip if goes over)
            custom_downsample (int): additionally downscale the heatmap by specified factor
            cmap (str): name of matplotlib colormap to use
        """
        # 确定可视化的层级，如果未指定，则选择最适合的层级
        if vis_level < 0:
            vis_level = self.wsi.get_best_level_for_downsample(32)
        # 获取当前层级的缩放比例
        downsample = self.level_downsamples[vis_level]
        # 从层级0到目标层级的缩放比例
        scale = [1 / downsample[0], 1 / downsample[1]]

        # 如果分数是二维数组，则将其展平成一维数组
        if len(scores.shape) == 2:
            scores = scores.flatten()

        # 如果需要二值化，则设置阈值
        if binarize:
            if thresh < 0:
                threshold = 1.0 / len(scores)
            else:
                threshold = thresh
        else:
            threshold = 0.0

        # 计算热图的大小并过滤指定bbox区域之外的坐标/分数
        if top_left is not None and bot_right is not None:
            scores, coords = screen_coords(scores, coords, top_left, bot_right)
            coords = coords - top_left
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
        else:
            region_size = self.level_dim[vis_level]
            top_left = (0, 0)
            bot_right = self.level_dim[0]

        # 根据缩放比例调整补丁大小和坐标
        # patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        coords = np.ceil(coords * np.array(scale)).astype(int)

        # 归一化过滤分数
        if convert_to_percentiles:
            scores = to_percentiles(scores)
        scores /= 100

        # 绘制前 k 个区域在原图上方便查看
        original_image = self.wsi.read_region(top_left, vis_level, region_size).convert("RGB")
        # sorted_scores_and_coords = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        # 分离排序后的分数和坐标
        # sorted_scores = [score for idx, score in sorted_scores_and_coords]
        # sorted_coords = [coords[idx] for idx, score in sorted_scores_and_coords]
        for idx, (s_coord, s_score) in enumerate(zip(coords, scores)):
            # 绘制矩形框和编号的坐标
            rectangle_coords = (tuple(s_coord), tuple(s_coord + patch_size))
            original_image = self.draw_rectangle_with_text(original_image, rectangle_coords, f'top: {idx + 1}')

        # 计算原始注意力得分的热图（在颜色图之前） 通过在重叠区域上累积分数
        # heatmap overlay: 跟踪热图每个像素的注意力得分
        overlay = np.full(np.flip(region_size), 0).astype(float)
        # overlay counter: 跟踪热图的每个像素上注意力得分的累积次数
        counter = np.full(np.flip(region_size), 0).astype(np.uint16)
        count = 0
        for idx in range(len(coords)):
            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:
                if binarize:
                    score = 1.0
                    count += 1
            else:
                score = 0.0
            # 累积注意力
            overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += score
            # 累积计数器
            counter[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += 1

        if binarize:
            print('\nbinarized tiles based on cutoff of {}'.format(threshold))
            print('identified {}/{} patches as positive'.format(count, len(coords)))

        # 获取关注区域和平均累积注意力
        zero_mask = counter == 0
        if binarize:
            overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
        else:
            overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
        del counter

        if blur:
            overlay = cv2.GaussianBlur(overlay, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

        # 组织分割掩码
        if segment:
            tissue_mask = self.get_seg_mask(region_size, scale, use_holes=use_holes, offset=tuple(top_left))

        # 如果不使用空白画布，则对原始图像进行下采样并用作画布
        if not blank_canvas:
            # 对原始图像进行下采样并用作画布
            img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
        else:
            # 使用空白画布
            img = np.array(Image.new(size=region_size, mode="RGB", color=(255, 255, 255)))

        # print('\ncomputing heatmap image')
        # print('total of {} patches'.format(len(coords)))
        # twenty_percent_chunk = max(1, int(len(coords) * 0.2))

        # 获取matplotlib颜色图
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        for idx in range(len(coords)):
            # if (idx + 1) % twenty_percent_chunk == 0:
            # print('progress: {}/{}'.format(idx, len(coords)))

            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:
                # 注意力块
                raw_block = overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]]
                # 图像块（空白画布或原始图像）
                img_block = img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]].copy()
                # 色块（cmap应用于注意力块）
                color_block = (cmap(raw_block) * 255)[:, :, :3].astype(np.uint8)

                # 将颜色块叠加到图像上
                cv2.addWeighted(img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]], alpha,
                                color_block, 1 - alpha, 0,
                                img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]])

                if segment:
                    # 组织掩码块
                    mask_block = tissue_mask[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]]
                    # 只复制组织掩码部分的颜色块
                    img_block[mask_block] = color_block[mask_block]
                else:
                    # 复制整个颜色块
                    img_block = color_block

                # 重写图像块
                img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] = img_block.copy()

        del overlay

        # 创建颜色条图像
        colorbar_fig = plt.figure(figsize=(1, 8))
        # 创建一个空白图像用于颜色条
        plt.imshow(np.zeros((256, 1)), cmap=cmap)
        # 添加颜色条
        plt.colorbar(orientation='vertical')
        # 隐藏坐标轴
        plt.axis('off')

        # 保存图像到BytesIO对象
        buffered = BytesIO()
        plt.savefig(buffered, format='png')
        # 关闭图形以释放资源
        plt.close(colorbar_fig)

        # 从BytesIO对象获取数据
        buffered.seek(0)
        # 使用Image打开图像
        colorbar_img = Image.open(buffered)

        # 对图像进行高斯模糊
        if blur:
            img = cv2.GaussianBlur(img, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

        # 如果alpha值小于1，则将热图与原始图像混合
        if alpha < 1.0:
            img = self.block_blending(img, vis_level, top_left, bot_right, alpha=alpha, blank_canvas=blank_canvas,
                                      block_size=1024)

        img = Image.fromarray(img)
        w, h = img.size

        # 对图像进行自定义下采样
        if custom_downsample > 1:
            img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

        # 如果图像尺寸超过最大尺寸，则进行缩放
        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size / w if w > h else max_size / h
            img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

        # 调整颜色条图像的大小以匹配主图像的宽度和高度
        # 假设我们想要颜色条的宽度为20像素，高度与热力图相同
        colorbar_width = 200  # 颜色条的宽度
        colorbar_img_resized = colorbar_img.resize((colorbar_width, img.height), Image.BILINEAR)

        # 调整颜色条图像的大小以匹配主图像的宽度和高度
        # colorbar_img_resized = colorbar_img.resize((colorbar_img.width, img.height))

        # 创建一个新的PIL图像来保存热图和颜色条
        # 确保新图像的宽度是主图像宽度和颜色条宽度之和，高度与主图像相同
        img_with_colorbar = Image.new('RGB', (img.width + colorbar_img_resized.width, img.height))

        # 将主图像粘贴到新图像的左侧
        img_with_colorbar.paste(img, (0, 0))

        # 将颜色条粘贴到新图像的右侧
        img_with_colorbar.paste(colorbar_img_resized, (img.width, 0))
        # 关闭BytesIO对象
        buffered.close()
        img = img_with_colorbar
        return img, original_image

    def block_blending(self, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
        # 获取当前层级的缩放比例
        downsample = self.level_downsamples[vis_level]

        # 获取图像的宽度和高度
        w = img.shape[1]
        h = img.shape[0]
        # 确定块的大小，不超过图像的宽或高
        block_size_x = min(block_size, w)
        block_size_y = min(block_size, h)
        # print('using block size: {} x {}'.format(block_size_x, block_size_y))

        # 计算偏移量， 相对于(0,0)的偏移量
        shift = top_left
        # 遍历图像区域，步长为块大小乘以缩放比例
        for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
            for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
                # 1. 通过平移和缩放将wsi坐标转换为图像坐标
                x_start_img = int((x_start - shift[0]) / int(downsample[0]))
                y_start_img = int((y_start - shift[1]) / int(downsample[1]))

                # 2. 计算混合块的结束点，注意不要超出图像边缘
                y_end_img = min(h, y_start_img + block_size_y)
                x_end_img = min(w, x_start_img + block_size_x)

                # 如果块的大小为0，则跳过
                if y_end_img == y_start_img or x_end_img == x_start_img:
                    continue

                # 3. 获取混合块和大小
                blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img]
                blend_block_size = (x_end_img - x_start_img, y_end_img - y_start_img)

                if not blank_canvas:
                    # 4. 将实际的wsi块读取为canvas块
                    pt = (x_start, y_start)
                    canvas = np.array(self.wsi.read_region(pt, vis_level, blend_block_size).convert("RGB"))
                else:
                    # 4. 创建空白画布块
                    canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255, 255, 255)))

                # 5. 混合色块和画布块
                img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas,
                                                                                    1 - alpha, 0, canvas)
        return img

    def get_seg_mask(self, region_size, scale, use_holes=False, offset=(0, 0)):
        # 创建一个与区域大小相同的掩码，初始值为0
        tissue_mask = np.full(np.flip(region_size), 0).astype(np.uint8)
        # 缩放组织轮廓的尺寸
        contours_tissue = self.scaleContourDim(self.contours_tissue, scale)
        # 计算偏移量
        offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))

        # 缩放空洞轮廓的尺寸
        contours_holes = self.scaleHolesDim(self.holes_tissue, scale)

        # 将组织轮廓和空洞轮廓按面积降序排序
        contours_tissue, contours_holes = zip(
            *sorted(zip(contours_tissue, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True))

        # 遍历轮廓，绘制组织掩码
        for idx in range(len(contours_tissue)):
            cv2.drawContours(image=tissue_mask, contours=contours_tissue, contourIdx=idx, color=(1), offset=offset,
                             thickness=-1)

            # 如果使用空洞，则绘制空洞轮廓
            if use_holes:
                cv2.drawContours(image=tissue_mask, contours=contours_holes[idx], contourIdx=-1, color=(0),
                                 offset=offset, thickness=-1)
            # contours_holes = self._scaleContourDim(self.holes_tissue, scale, holes=True, area_thresh=area_thresh)

        tissue_mask = tissue_mask.astype(bool)
        return tissue_mask

    def draw_rectangle_with_text(self, image, coord, text, font_path=None, font_size=20):
        """
        在图像上绘制矩形框和文本。
        :param image: PIL图像对象。
        :param coord: 矩形左上角和右下角的坐标，格式为((x1, y1), (x2, y2))。
        :param text: 要绘制的文本。
        :param font_path: 字体文件的路径。如果为None，则使用默认字体。
        :param font_size: 字体大小。
        """
        draw = ImageDraw.Draw(image)

        # 绘制矩形框
        draw.rectangle(coord, outline='red', width=5)

        # 加载字体
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        # 使用ImageDraw对象的getsize方法获取文本尺寸
        bbox = draw.textbbox((0, 0), text, font=font)

        # 计算文本的中心位置
        x1, y1 = coord[0]
        x2, y2 = coord[1]
        text_x = x1 + (x2 - x1 - (bbox[2] - bbox[0])) / 2
        text_y = y1 + (y2 - y1 - (bbox[3] - bbox[1])) / 2

        # 绘制文本
        draw.text((text_x, text_y), text, fill='red', font=font)

        # 返回修改后的图像
        return image
