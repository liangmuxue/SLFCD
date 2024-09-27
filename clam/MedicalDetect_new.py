import sys
sys.path.append("/home/bavon/project/SLFCD/SLFCD/extras/")
sys.path.append("/home/bavon/project/SLFCD/SLFCD/project/")
sys.path.append("/home/bavon/project/SLFCD/SLFCD/")

import stat
import shutil
import os
import argparse
from clam.models.model_clam2 import Classifier
import yaml
from clam.utils.file_utils import save_hdf5
from tqdm import tqdm
from wsi_core.InferSVSProcess import WholeSlideImage
from clam.datasets.dataset_h5 import Whole_Slide_Bag_FP_all

from multiprocessing import Queue
import json
import threading
from threading import Thread
import requests
import urllib.request
from flask import Flask, request, jsonify
import cv2
from PIL import Image, ImageDraw
import numpy as np
import h5py
import random

from shapely.geometry import box
from BoxDeduplication import merge_overlapping_rectangles, get_vertices

import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader
from ultralytics.nn.autobackend import AutoBackend

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='inference script')
parser.add_argument('--data_dir', type=str, default=r'D:\project\SLFCD\dataset\ais\data\1-CG23_18831_01.svs',
                    help='svs file')
parser.add_argument('--save_path', type=str, default="heatmaps/output")
parser.add_argument('--root_url', type=str, default="192.168.0.98")
parser.add_argument('--root_port', type=int, default=8088)
parser.add_argument('--receive_port', type=str, default=8091)
parser.add_argument('--config_file', type=str, default="infer2.yaml")
parser.add_argument('--device', type=str, default="cuda:1")
args = parser.parse_args()
root_url, root_port, receive_port = args.root_url, args.root_port, args.receive_port

svs_input_queue = Queue(6)
svs_output_queue_15 = Queue(6)
stage_send_queue_multi = Queue(6)
svs_output_queue_all = Queue(6)
queues_in = Queue()
queues_out = Queue()
stop_Process_dict = {}
stop_event = threading.Event()
dzi_event = threading.Event()

stage_flag = {'ais': 1, 'hsil': 2, 'lsil': 3, 1: 'ais模型推理完成', 2: "hsil模型推理完成", 3: "lsil模型推理完成",
              4: '样本模糊无效'}

infer_thread = None

app = Flask(__name__)


class LetterBox:
    def __init__(self, new_shape=(640, 640), auto=False, center=True, stride=32, half=False, device='cpu'):
        self.new_shape = new_shape
        self.auto = auto
        self.stride = stride
        self.half = half
        self.center = center  # Put the image in the middle or top-left
        self.device = device

    def __call__(self, image):
        if isinstance(image,  Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = np.asarray(image)[:, :, ::-1]
            image = np.ascontiguousarray(image)  # contiguous

        shape = image.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

        im = np.stack([image])
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)
        im = im.to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        return im


class YOLOInfer:
    def __init__(self, auto, stride, half, device):
        self.letterbox = LetterBox(auto=auto, stride=stride, half=half, device=device)

    def xywh2xyxy(self, x):
        assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
        y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
        dw = x[..., 2] / 2  # half-width
        dh = x[..., 3] / 2  # half-height
        y[..., 0] = x[..., 0] - dw  # top left x
        y[..., 1] = x[..., 1] - dh  # top left y
        y[..., 2] = x[..., 0] + dw  # bottom right x
        y[..., 3] = x[..., 1] + dh  # bottom right y
        return y

    def clip_boxes(self, boxes, shape):
        if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
            boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
            boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
            boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
            boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
        return boxes

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (
                round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
                round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
            )  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        if padding:
            boxes[..., 0] -= pad[0]  # x padding
            boxes[..., 1] -= pad[1]  # y padding
            if not xywh:
                boxes[..., 2] -= pad[0]  # x padding
                boxes[..., 3] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        return self.clip_boxes(boxes, img0_shape)

    def non_max_suppression(self, prediction, inferShape, orgShape, conf_thres=0.25, iou_thres=0.45, agnostic=True,
                            max_wh=7680, nc=0):
        nc = nc  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

        prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
        prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])  # xywh to xyxy

        true_indices = torch.nonzero(xc)
        selected_rows = prediction[true_indices[:, 0], true_indices[:, 1]]
        new_prediction = torch.cat((selected_rows, true_indices[:, 0].unsqueeze(1).float()), dim=1)

        if new_prediction.shape[0] == 0:
            return torch.zeros((0, 5))

        box, cls, mask, idxs = new_prediction.split((4, nc, nm, 1), 1)
        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.squeeze(-1) > conf_thres]
        if not x.shape[0]:  # no boxes
            return torch.zeros((0, 5))

        # cls = x[:, 5]  # classes
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        idxs = idxs.t().squeeze(0)

        keep = torchvision.ops.batched_nms(boxes, scores, idxs, iou_thres)
        boxes[keep] = self.scale_boxes(inferShape, boxes[keep], orgShape)

        boxes = boxes[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()
        # cls = cls[keep].cpu().numpy()
        # idxs = idxs[keep].cpu().numpy()

        results = np.hstack((boxes, np.expand_dims(scores, axis=1)))
        # results = np.hstack((results, np.expand_dims(cls, axis=1)))
        # results = np.hstack((results, np.expand_dims(idxs, axis=1)))
        return results

    def is_almost_contained(self, box1, box2, threshold=0.8):
        """
        检查 box2 是否几乎完全包含在 box1 内
        box1 和 box2 是矩形框的坐标，格式为 [x1, y1, x2, y2]
        threshold 是包含的最小比例阈值
        """
        # 计算交集区域的坐标
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
    
        # 计算交集区域的面积
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
        # 如果交集区域面积为0，说明没有交集
        if intersection_area == 0:
            return False
    
        # 计算 box2 的面积
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
        # 计算 box2 与交集区域面积的比例
        containment_ratio = intersection_area / box2_area
    
        return containment_ratio > threshold

    def intersection_over_union(self, boxA, boxB):
        # 计算两个边界框的交并比(IOU)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
    
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def non_minmax_suppression(self, boxes, scores, iou_threshold=0.5):
        """
        实现非极大值抑制(NMS)，输入是边界框和对应的分数，
        返回经过NMS处理后的边界框列表。
        """
        # 根据分数排序
        sorted_indices = np.argsort(scores)[::-1]
    
        keep_boxes = []
        while sorted_indices.size > 0:
            # 选择当前最高分的框
            idx = sorted_indices[0]
            keep_boxes.append(idx)
    
            # 计算当前框与其他所有框的IOU
            ious = np.array([self.intersection_over_union(boxes[idx], boxes[i]) for i in sorted_indices[1:]])
    
            # 删除与当前框IOU大于阈值的框
            remove_indices = np.where(ious > iou_threshold)[0] + 1  # +1是因为我们忽略了第一个元素（当前最高分的框）
            sorted_indices = np.delete(sorted_indices, remove_indices)
            sorted_indices = np.delete(sorted_indices, 0)  # 移除已经处理过的最高分框的索引
    
        keep_boxes = boxes[keep_boxes].tolist()
        while True:
            keep_boxes = sorted(keep_boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
            all_keep = []
            merged_boxes = []
            while keep_boxes:
                current = keep_boxes.pop(0)
                all_keep.append(current)
                for box in keep_boxes:
                    if self.is_almost_contained(current, box, threshold=0):
                        merged_box = [min(current[0], box[0]), min(current[1], box[1]),
                                      max(current[2], box[2]), max(current[3], box[3])]
                        if merged_box not in merged_boxes:
                            merged_boxes.append(merged_box)
                        keep_boxes.remove(box)
                        if current in all_keep:
                            all_keep.remove(current)
    
            keep_boxes = all_keep + merged_boxes
            if not merged_boxes or len(keep_boxes) == 1:
                break
    
        return keep_boxes


class Infer(Thread):
    def __init__(self, args):
        super().__init__()
        config_path = os.path.join('heatmaps/configs', args.config_file)
        config_dict = yaml.safe_load(open(config_path, 'r'))
        config_dict["data_arguments"]["data_dir"] = args.data_dir
        config_dict["data_arguments"]["save_path"] = args.save_path
        config_dict["model_arguments"]["device"] = args.device

        self.data_args = argparse.Namespace(**config_dict['data_arguments'])
        self.model_args = argparse.Namespace(**config_dict['model_arguments'])

        self.ais_params = argparse.Namespace(**config_dict['ais_arguments'])
        self.hsil_params = argparse.Namespace(**config_dict['hsil_arguments'])
        self.lsil_params = argparse.Namespace(**config_dict['lsil_arguments'])

        self.def_seg_params = config_dict['seg_arguments']
        self.def_filter_params = config_dict['filter_arguments']
        self.def_vis_params = config_dict['vis_arguments']
        self.def_patch_params = config_dict['patch_arguments']

        self.custom_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224)])

        self.cls_label = {0: 'normal', 1: 'ais', 2: 'hsil', 3: "lsil"}

        # ----------------------------- load model ------------------------------
        self.cls_model_1, self.cls_model_2, self.model_ais, self.model_hsil, self.model_lsil = self.load_model()

        self.results = {}
        self.orig_results = {
            'zoom': [],
            "category": "normal",
            "boxes": {
                "ais": [],
                "hsil": [],
                "lsil": []
            },
            'size': {"ais": [],
                     "hsil": [],
                     "lsil": []
                     },
            'level': {'ais': 1,
                      'hsil': 1,
                      'lsil': 0}
        }

        self.results_0 = {}
        self.orig_results_0 = {
            "boxes": {
                "ais": [],
                "hsil": [],
                "lsil": []
            },
            'size': [],
            'level': {'ais': 1,
                      'hsil': 1,
                      'lsil': 0
                      },
            "box_point_vertices": {
                "ais": [],
                "hsil": [],
                "lsil": []
            }
        }
    
    def is_thread_started(self):
        return self.is_alive()
    
    def load_model(self):
        # resnet18
        cls_model_1 = models.resnet18(pretrained=False)
        for param in cls_model_1.parameters():
            param.requires_grad = False
        cls_model_1.fc = nn.Linear(cls_model_1.fc.in_features, 2048)
        cls_model_1 = cls_model_1.to(self.model_args.device)
        model_state_dict = torch.load(self.model_args.cls_ckpt_path_1)
        state_dict = {k: v for k, v in model_state_dict.items() if 'fc' not in k}
        cls_model_1.load_state_dict(state_dict, strict=False)
        cls_model_1.to(args.device)
        cls_model_1.eval()
        print('load cls ckpt path 1: {}'.format(self.model_args.cls_ckpt_path_1))

        # cls_model
        cls_model_2 = Classifier(2048, self.model_args.n_classes)
        ckpt = torch.load(self.model_args.cls_ckpt_path_2, map_location=self.model_args.device)
        cls_model_2.load_state_dict(ckpt, strict=False)
        cls_model_2.to(args.device)
        cls_model_2.eval()
        print('load cls ckpt path 2: {}'.format(self.model_args.cls_ckpt_path_2))

        # ais
        model_ais = AutoBackend(self.model_args.ais_ckpt_path, torch.device(self.model_args.device))
        model_ais.eval()
        print('load ais ckpt path: {}'.format(self.model_args.ais_ckpt_path))

        # hsil
        model_hsil = AutoBackend(self.model_args.hsil_ckpt_path, torch.device(self.model_args.device))
        model_hsil.eval()
        print('load hsil ckpt path: {}'.format(self.model_args.hsil_ckpt_path))

        # lsil
        model_lsil = AutoBackend(self.model_args.lsil_ckpt_path, torch.device(self.model_args.device))
        model_lsil.eval()
        print('first lsil ckpt path: {}'.format(self.model_args.lsil_ckpt_path))

        return cls_model_1, cls_model_2, model_ais, model_hsil, model_lsil

    def process_svs(self, save_path, svs_path, svs_id, seg_params, patch_level, patch_size, step_size, name=None,
                    flag=False):
        slide_id = os.path.split(svs_path)[-1][:-4]

        # ----------------------- save -------------------------------------------
        root_path = os.path.join(save_path, svs_id)
        if flag:
            if not os.path.exists(root_path):
                os.makedirs(root_path, exist_ok=True)
                os.chmod(root_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

            save_path_hdf5 = os.path.join(root_path, svs_id + ".h5")
            pt_save_path = os.path.join(root_path, svs_id + '_cls.pt')
            h5_save_path = os.path.join(root_path, svs_id + '_cls.h5')
        else:
            slide_save_dir = os.path.join(save_path, svs_id, name)
            os.makedirs(slide_save_dir, exist_ok=True)
            os.chmod(slide_save_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            save_path_hdf5 = os.path.join(slide_save_dir, slide_id + ".h5")

            small_img_15 = os.path.join(slide_save_dir, 'smallPic15')
            if os.path.exists(small_img_15):
                shutil.rmtree(small_img_15)
            os.makedirs(small_img_15, exist_ok=True)
            os.chmod(small_img_15, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

            small_img_All = os.path.join(slide_save_dir, 'smallPicAll')
            if os.path.exists(small_img_All):
                shutil.rmtree(small_img_All)
            os.makedirs(small_img_All, exist_ok=True)
            os.chmod(small_img_All, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

            infer_box_img = os.path.join(slide_save_dir, f'{svs_id}.jpg')
            full_img = os.path.join(root_path, f'full_{svs_id}.jpg')

        # Load segmentation and filter parameters
        seg_params['seg_level'] = patch_level
        filter_params = self.def_filter_params.copy()
        patch_params = self.def_patch_params.copy()
        patch_params.update({'patch_level': patch_level, 'patch_size': patch_size,
                             'step_size': step_size, 'save_path_hdf5': save_path_hdf5, 'svs_id': svs_id})

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []
        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        WSI_object = WholeSlideImage(svs_path, patch_level)
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

        if WSI_object.contours_tissue:
            WSI_object.process_contours(**patch_params)
        if flag:
            return WSI_object, save_path_hdf5, pt_save_path, h5_save_path
        else:
            return WSI_object, save_path_hdf5, full_img, infer_box_img, small_img_15, small_img_All


    def cls_svs(self, file_id, file_path, WSI_object, patch_level, patch_size, step_size, pt_save_path, h5_save_path,
                stop_event):
        cls_flag = False
        dataset = Whole_Slide_Bag_FP_all(file_path=file_path, WSI_object=WSI_object,
                                         patch_level=patch_level, patch_size=patch_size, slide_size=step_size,
                                         transforms=self.custom_transforms)

        loader = DataLoader(dataset=dataset, batch_size=self.model_args.batch_size, pin_memory=True,
                            num_workers=self.model_args.num_workers)

        mode = 'w'
        with torch.no_grad():
            for batch, coords in tqdm(loader, total=len(loader), desc=f"cls infer {file_id}"):
                if stop_event.is_set():
                    cls_flag = True
                    break
                batch = batch.to(self.model_args.device)

                features = self.cls_model_1(batch)

                asset_dict = {'features': features.cpu().numpy(), 'coords': coords.cpu().numpy()}
                save_hdf5(h5_save_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'

            if cls_flag:
                return None
            file = h5py.File(h5_save_path, "r")
            features = file['features'][:]
            features = torch.from_numpy(features)
            torch.save(features, pt_save_path)
            # features = torch.cat([item for item in features], dim=1)
            features = features.to(self.model_args.device)

            logits, _, Y_hat = self.cls_model_2(features)
            print(logits, Y_hat, self.cls_label[Y_hat.cpu().item()])
            return self.cls_label[Y_hat.cpu().item()]

    def infer(self, name, svs_id, file_path, WSI_object, yolo_infer, patch_level, img_size, patch_size, slide_size, model, stop_event, 
              stage_send_queue_multi, full_img, small_img_15, small_img_All, box_img_save):
        infer_flag = False

        image, scale, shape = WSI_object.img, WSI_object.scale, WSI_object.shape
        image_rgb = image.convert('RGB')
        slide_length = patch_size // slide_size
        image_rgb.save(full_img)

        with h5py.File(file_path, "a") as f:
            coords = np.array(f['coords'])
            repeat_coord = []
            big_image_boxes = []
            for coord in tqdm(coords, total=coords.shape[0], desc=f"{name} infer {svs_id}"):
                if stop_event.is_set():
                    infer_flag = True
                    break
                coord = np.array(coord / scale).astype(int)
                for i in range(slide_length):
                    for j in range(slide_length):
                        # 计算切分区域的坐标
                        coord_tar = np.array([coord[0] + i * slide_size, coord[1] + j * slide_size]).astype(np.int16)
                        coord_tar = [coord_tar[0], coord_tar[1], coord_tar[0] + img_size, coord_tar[1] + img_size]
                        if coord_tar[2] < shape[0] and coord_tar[3] < shape[1]:
                            if coord_tar not in repeat_coord:
                                repeat_coord.append(coord_tar)

                                x1, y1, x2, y2 = coord_tar
                                new_image = image_rgb.crop((x1, y1, x2, y2))

                                im = yolo_infer.letterbox(new_image)
                                results = model(im, augment=False)[0]
                                results = yolo_infer.non_max_suppression(results, im.shape[2:], new_image.size, 0.5, 0.45, nc=1)

                                for xyxy, conf in zip(results[:, :4].tolist(), results[:, 4].tolist()):
                                    # 将坐标转换为大图的坐标
                                    x1_big = xyxy[0] + x1
                                    y1_big = xyxy[1] + y1
                                    x2_big = xyxy[2] + x1
                                    y2_big = xyxy[3] + y1
                                    # 确保坐标在大图的边界内
                                    x1_big = int(max(0, min(x1_big, shape[0])))
                                    y1_big = int(max(0, min(y1_big, shape[1])))
                                    x2_big = int(max(0, min(x2_big, shape[0])))
                                    y2_big = int(max(0, min(y2_big, shape[1])))
                                    # 存储或处理转换后的坐标
                                    big_image_boxes.append([x1_big, y1_big, x2_big, y2_big, conf])

        if infer_flag:
            return None

        if big_image_boxes:
            big_image_boxes = np.array(big_image_boxes)
            big_image_boxes = yolo_infer.non_minmax_suppression(big_image_boxes[:, :4], big_image_boxes[:, 4], iou_threshold=0.5)
            
            total_img_copy1 = image_rgb.copy()
            total_img_copy2 = image_rgb.copy()
            draw = ImageDraw.Draw(total_img_copy2)
            
            coord_15, coord_all, box_point_vertices = [], [], []
            for region in tqdm(big_image_boxes, total=len(big_image_boxes), desc=f"save img {name} {svs_id}"):
                region = [int(i) for i in region]
                draw.rectangle(region, outline="red", width=2)
                                                    
                img_ori = total_img_copy1.crop(region)
                if len(os.listdir(small_img_15)) < 15:
                    img_ori.save(f"{small_img_15}/[{region[0]},{region[1]},{region[2]},{region[3]}].jpg")
                    coord_15.append(region)

                img_ori.save(f"{small_img_All}/[{region[0]},{region[1]},{region[2]},{region[3]}].jpg")
                coord_all.append(region)

            total_img_copy2.save(box_img_save)
            
            rectangles = [box(*i) for i in coord_all]
            merged_rectangles = merge_overlapping_rectangles(rectangles)
            box_point_vertices = get_vertices(merged_rectangles)

        else:
            coord_15, coord_all, box_point_vertices = [], [], []

        if not stage_send_queue_multi.empty():
            stage_send_queue_multi.get()
        print(f"stage {stage_flag[name]} {stage_flag[stage_flag[name]]}")
        print(f"name: {name}, sampled_coords: {coord_15}, box_point_vertices: {box_point_vertices}")
        stage_send_queue_multi.put({'sampleId': svs_id, 'state': stage_flag[name]})
        return {
            name: {'coord_15': coord_15, "coord_all": coord_all, "size": WSI_object.wsi.level_dimensions[patch_level],
                   'zoom': WSI_object.wsi.level_downsamples, 'level': WSI_object.wsi.level_downsamples[patch_level],
                   'box_point_vertices': box_point_vertices}}

    def run(self):
        self.img_size = self.model_args.img_size
        self.patch_size = self.model_args.patch_size
        self.slide_size = self.model_args.slide_size
        
        # ----------------------------- ais ------------------------------
        self.patch_level_ais = self.ais_params.patch_level
        
        # ----------------------------- hsil ------------------------------
        self.patch_level_hsil = self.hsil_params.patch_level

        # ----------------------------- lsil ------------------------------
        self.patch_level_lsil = self.lsil_params.patch_level

        yolo_infer = YOLOInfer(self.model_ais.pt, self.model_ais.stride, self.model_ais.fp16, self.model_args.device)

        while True:
            while not stop_event.is_set():
                results = {}

                svs_content = queues_in.get()

                save_url, file_path, file_id = svs_content

                self.results["infer_id"] = file_id
                self.results[file_id] = self.orig_results

                self.results_0["infer_id"] = file_id
                self.results_0[file_id] = self.orig_results_0

                # ------------------ 对 SVS 进行分类检测的前处理 ------------------
                if stop_event.is_set():
                    break
                seg_params = self.def_seg_params.copy()
                WSI_object, save_path_hdf5, \
                    pt_save_path, h5_save_path = self.process_svs(save_url, file_path, file_id, seg_params,
                                                                  self.model_args.cls_arguments["patch_level"],
                                                                  self.model_args.cls_arguments["patch_size"],
                                                                  self.model_args.cls_arguments["step_size"],
                                                                  flag=True)

                # ------------------ 对 SVS 进行分类检测的推理处理 ------------------
                if stop_event.is_set():
                    break
                if os.path.exists(save_path_hdf5):
                    infer_category = self.cls_svs(file_id, save_path_hdf5, WSI_object,
                                                  self.model_args.cls_arguments["patch_level"],
                                                  self.model_args.cls_arguments["patch_size"],
                                                  self.model_args.cls_arguments["step_size"],
                                                  pt_save_path, h5_save_path, stop_event)

                    self.results[file_id]['category'] = infer_category

                    if stop_event.is_set():
                        break

                    if infer_category == 'ais':
                        # ----------------------------- ais infer ------------------------------
                        seg_params = self.def_seg_params.copy()
                        WSI_object, save_path_hdf5, full_img, infer_box_img, \
                            small_img_15, small_img_All = self.process_svs(save_url, file_path, file_id, seg_params,
                                                                           self.patch_level_ais,
                                                                           self.patch_size,
                                                                           self.slide_size,
                                                                           name='ais')

                        if stop_event.is_set():
                            break
                        result_ais = self.infer('ais', file_id, save_path_hdf5, WSI_object, yolo_infer, self.patch_level_ais, 
                                                self.img_size, self.patch_size, self.slide_size,
                                                self.model_ais, stop_event, stage_send_queue_multi,
                                                 full_img, small_img_15, small_img_All, infer_box_img)

                        if stop_event.is_set():
                            break

                        results.update(result_ais)

                        # ----------------------------- hsil infer ------------------------------
                        seg_params = self.def_seg_params.copy()
                        WSI_object, save_path_hdf5, full_img, infer_box_img, \
                            small_img_15, small_img_All = self.process_svs(save_url, file_path, file_id, seg_params,
                                                                           self.patch_level_hsil,
                                                                           self.patch_size,
                                                                           self.slide_size,
                                                                           name='hsil')

                        if stop_event.is_set():
                            break

                        result_hsil = self.infer('hsil', file_id, save_path_hdf5, WSI_object, yolo_infer, self.patch_level_hsil, 
                                                 self.img_size, self.patch_size, self.slide_size,
                                                 self.model_hsil, stop_event, stage_send_queue_multi,
                                                 full_img, small_img_15, small_img_All, infer_box_img)
                        
                        if stop_event.is_set():
                            break
                        results.update(result_hsil)

                        # ----------------------------- lsil infer ------------------------------
                        seg_params = self.def_seg_params.copy()
                        WSI_object, save_path_hdf5, full_img, infer_box_img, \
                            small_img_15, small_img_All = self.process_svs(save_url, file_path, file_id, seg_params,
                                                                           self.patch_level_lsil,
                                                                           self.patch_size,
                                                                           self.slide_size,
                                                                           name='lsil')

                        if stop_event.is_set():
                            break

                        result_lsil = self.infer('lsil', file_id, save_path_hdf5, WSI_object, yolo_infer, self.patch_level_lsil, 
                                                 self.img_size, self.patch_size, self.slide_size,
                                                 self.model_lsil, stop_event, stage_send_queue_multi,
                                                 full_img, small_img_15, small_img_All, infer_box_img)
                        
                        if stop_event.is_set():
                            break
                        results.update(result_lsil)

                    elif infer_category == 'hsil':
                        # ----------------------------- hsil infer ------------------------------
                        seg_params = self.def_seg_params.copy()
                        WSI_object, save_path_hdf5, full_img, infer_box_img, \
                            small_img_15, small_img_All = self.process_svs(save_url, file_path, file_id, seg_params,
                                                                           self.patch_level_hsil,
                                                                           self.patch_size,
                                                                           self.slide_size,
                                                                           name='hsil')

                        if stop_event.is_set():
                            break

                        result_hsil = self.infer('hsil', file_id, save_path_hdf5, WSI_object, yolo_infer, self.patch_level_hsil, 
                                                 self.img_size, self.patch_size, self.slide_size,
                                                 self.model_hsil, stop_event, stage_send_queue_multi,
                                                 full_img, small_img_15, small_img_All, infer_box_img)
                        if stop_event.is_set():
                            break
                        results.update(result_hsil)

                        # ----------------------------- lsil infer ------------------------------
                        seg_params = self.def_seg_params.copy()
                        WSI_object, save_path_hdf5, full_img, infer_box_img, \
                            small_img_15, small_img_All = self.process_svs(save_url, file_path, file_id, seg_params,
                                                                           self.patch_level_lsil,
                                                                           self.patch_size,
                                                                           self.slide_size,
                                                                           name='lsil')

                        if stop_event.is_set():
                            break

                        result_lsil = self.infer('lsil', file_id, save_path_hdf5, WSI_object, yolo_infer, self.patch_level_lsil, 
                                                 self.img_size, self.patch_size, self.slide_size,
                                                 self.model_lsil, stop_event, stage_send_queue_multi,
                                                 full_img, small_img_15, small_img_All, infer_box_img)
                        if stop_event.is_set():
                            break
                        results.update(result_lsil)

                    elif infer_category == 'lsil':
                        # ----------------------------- lsil infer ------------------------------
                        seg_params = self.def_seg_params.copy()
                        WSI_object, save_path_hdf5, full_img, infer_box_img, \
                            small_img_15, small_img_All = self.process_svs(save_url, file_path, file_id, seg_params,
                                                                           self.patch_level_lsil,
                                                                           self.patch_size,
                                                                           self.slide_size,
                                                                           name='lsil')

                        if stop_event.is_set():
                            break

                        result_lsil = self.infer('lsil', file_id, save_path_hdf5, WSI_object, yolo_infer, self.patch_level_lsil, 
                                                 self.img_size, self.patch_size, self.slide_size,
                                                 self.model_lsil, stop_event, stage_send_queue_multi,
                                                 full_img, small_img_15, small_img_All, infer_box_img)
                        if stop_event.is_set():
                            break
                        results.update(result_lsil)

                    elif infer_category == 'normal':
                        results = {
                            'ais': {'coord_15': [], 'coord_all': [], 'size': (9134, 7666), 'zoom': (4, 4), 'level': 4,
                                    'box_point_vertices': []},
                            'hsil': {'coord_15': [], 'coord_all': [], 'size': (9134, 7666), 'zoom': (4, 4), 'level': 4,
                                     'box_point_vertices': []},
                            'lsil': {'coord_15': [], 'coord_all': [], 'size': (9134, 7666), 'zoom': (1, 1), 'level': 1,
                                     'box_point_vertices': []}}

                try:
                    self.process_result(file_id, results)

                    while True:
                        if not dzi_event.is_set():
                            svs_output_queue_15.put(self.results)
                            svs_output_queue_all.put(self.results_0)
                            break

                    if not stage_send_queue_multi.empty():
                        stage_send_queue_multi.get()
                    stage_send_queue_multi.put({'sampleId': file_id, 'state': 4})
                    print("stage 4 推理结果处理完成，样本有效")
                    
                except Exception as e:
                    if not stage_send_queue_multi.empty():
                        stage_send_queue_multi.get()
                    print("stage 5 推理结果处理完成，样本模糊无效", e)
                    stage_send_queue_multi.put({'sampleId': file_id, 'state': 5})
                stop_event.set()

    def process_result(self, file_id, results):
        if self.results[file_id]['category'] == 'ais':
            self.results[file_id]['zoom'] = [int(i) for i in results.get('ais')['zoom']]

            self.results[file_id]['boxes']['ais'] = results.get('ais')['coord_15']
            self.results[file_id]['boxes']['hsil'] = results.get('hsil')['coord_15']
            self.results[file_id]['boxes']['lsil'] = results.get('lsil')['coord_15']

            self.results[file_id]['size']['ais'] = [int(i) for i in results.get('ais')['size']]
            self.results[file_id]['size']['hsil'] = [int(i) for i in results.get('hsil')['size']]
            self.results[file_id]['size']['lsil'] = [int(i) for i in results.get('lsil')['size']]

            self.results[file_id]['level']['ais'] = int(results.get('ais')['level'])
            self.results[file_id]['level']['hsil'] = int(results.get('hsil')['level'])
            self.results[file_id]['level']['lsil'] = int(results.get('lsil')['level'])

            self.results_0[file_id]['size'] = [int(i) for i in results.get('lsil')['size']]
            self.results_0[file_id]['boxes']['ais'] = results.get('ais')['coord_all']
            self.results_0[file_id]['boxes']['hsil'] = results.get('hsil')['coord_all']
            self.results_0[file_id]['boxes']['lsil'] = results.get('lsil')['coord_all']
            self.results_0[file_id]['level']['ais'] = int(results.get('ais')['level'])
            self.results_0[file_id]['level']['hsil'] = int(results.get('hsil')['level'])
            self.results_0[file_id]['level']['lsil'] = int(results.get('lsil')['level'])
            self.results_0[file_id]['box_point_vertices']['ais'] = results.get('ais')['box_point_vertices']
            self.results_0[file_id]['box_point_vertices']['hsil'] = results.get('hsil')['box_point_vertices']
            self.results_0[file_id]['box_point_vertices']['lsil'] = results.get('lsil')['box_point_vertices']

        elif self.results[file_id]['category'] == 'hsil':
            self.results[file_id]['zoom'] = [int(i) for i in results.get('hsil')['zoom']]

            self.results[file_id]['boxes']['hsil'] = results.get('hsil')['coord_15']
            self.results[file_id]['boxes']['lsil'] = results.get('lsil')['coord_15']

            self.results[file_id]['size']['hsil'] = [int(i) for i in results.get('hsil')['size']]
            self.results[file_id]['size']['lsil'] = [int(i) for i in results.get('lsil')['size']]

            self.results[file_id]['level']['hsil'] = int(results.get('hsil')['level'])
            self.results[file_id]['level']['lsil'] = int(results.get('lsil')['level'])

            self.results_0[file_id]['size'] = [int(i) for i in results.get('lsil')['size']]
            self.results_0[file_id]['boxes']['hsil'] = results.get('hsil')['coord_all']
            self.results_0[file_id]['boxes']['lsil'] = results.get('lsil')['coord_all']
            self.results_0[file_id]['level']['hsil'] = int(results.get('hsil')['level'])
            self.results_0[file_id]['level']['lsil'] = int(results.get('lsil')['level'])
            self.results_0[file_id]['box_point_vertices']['hsil'] = results.get('hsil')['box_point_vertices']
            self.results_0[file_id]['box_point_vertices']['lsil'] = results.get('lsil')['box_point_vertices']

        elif self.results[file_id]['category'] == 'lsil':
            self.results[file_id]['zoom'] = [int(i) for i in results.get('lsil')['zoom']]

            self.results[file_id]['boxes']['lsil'] = results.get('lsil')['coord_15']

            self.results[file_id]['size']['lsil'] = [int(i) for i in results.get('lsil')['size']]

            self.results[file_id]['level']['lsil'] = int(results.get('lsil')['level'])

            self.results_0[file_id]['size'] = [int(i) for i in results.get('lsil')['size']]
            self.results_0[file_id]['boxes']['lsil'] = results.get('lsil')['coord_all']
            self.results_0[file_id]['level']['lsil'] = int(results.get('lsil')['level'])
            self.results_0[file_id]['box_point_vertices']['lsil'] = results.get('lsil')['box_point_vertices']

        elif self.results[file_id]['category'] == 'normal':
            self.results[file_id]['zoom'] = []

            self.results[file_id]['boxes']['ais'] = []
            self.results[file_id]['boxes']['hsil'] = []
            self.results[file_id]['boxes']['lsil'] = []

            self.results[file_id]['size']['ais'] = []
            self.results[file_id]['size']['hsil'] = []
            self.results[file_id]['size']['lsil'] = []

            self.results[file_id]['level']['ais'] = 4
            self.results[file_id]['level']['hsil'] = 4
            self.results[file_id]['level']['lsil'] = 1

            self.results_0[file_id]['size'] = []
            self.results_0[file_id]['boxes']['ais'] = []
            self.results_0[file_id]['boxes']['hsil'] = []
            self.results_0[file_id]['boxes']['lsil'] = []
            self.results_0[file_id]['level']['ais'] = 4
            self.results_0[file_id]['level']['hsil'] = 4
            self.results_0[file_id]['level']['lsil'] = 1
            self.results_0[file_id]['box_point_vertices']['ais'] = []
            self.results_0[file_id]['box_point_vertices']['hsil'] = []
            self.results_0[file_id]['box_point_vertices']['lsil'] = []

infer_thread = Infer(args)
infer_thread.start()

@app.route('/upload_svs_file', methods=['POST'])
def upload_svs_file():
    if not infer_thread.is_thread_started():
        infer_thread.start()
    if infer_thread:
        pass
    try:
        data = request.get_json()
        save_url = data.get('saveUrl')
        file_url = data.get('svs_path')
        file_id = data.get('sampleId')
        print(save_url, file_url, file_id)

        if not os.path.exists(file_url):
            print("upload 1 文件不存在")
            return jsonify({'error': 1}), 404

        if not svs_output_queue_15.empty():
            results = svs_output_queue_15.get()
            if file_id in results:
                if svs_output_queue_15.empty():
                    svs_output_queue_15.put(results)
                print("upload 2 当前已有文件正在处理")
                return jsonify({'error': 2}), 200

        stop_event.clear()
        # 启动线程处理文件
        queues_in.put([save_url, file_url, file_id])

        process_dzi_thread = Thread(target=process_dzi_file, args=(save_url, file_url, file_id, dzi_event))
        process_dzi_thread.start()

        stage_thread = Thread(target=stage_send, args=(stage_send_queue_multi,))
        stage_thread.start()

        # 文件接收成功，返回提示信息
        print("upload 0 文件上传成功")
        return jsonify({'error': 0}), 200
    except Exception as e:
        print("upload 3 当前状态异常", e)
        return jsonify({'error': 3, 'message': e}), 500


def process_svs_file(infer, save_url, file_path, file_id, stage_send_queue_multi, svs_output_queue_15,
                     svs_output_queue_all, stop_event):
    if not stage_send_queue_multi.empty():
        stage_send_queue_multi.get()
    print("stage 0 文件正在分析")
    stage_send_queue_multi.put({'sampleId': file_id, 'state': 0})
    infer.main(save_url, file_path, file_id, stage_send_queue_multi, svs_output_queue_15, svs_output_queue_all,
               stop_event)


def process_dzi_file(save_url, file_path, file_id, dzi_event):
    dzi_event.set()
    data = {'saveUrl': save_url, 'svs_path': file_path, "sampleId": file_id}
    response = requests.post(f'http://192.168.0.179:8088/svs2dzi', json=data)
    dzi_event.clear()
    print("send_dzi: ", response.json())


def stage_send(stage_send_queue_multi):
    while True:
        try:
            # if not stage_send_queue_multi.empty():
            results = stage_send_queue_multi.get()
            req = urllib.request.Request(url=f'http://{root_url}:{receive_port}/system/job/stageSend',
                                         data=json.dumps(results).encode('utf-8'),
                                         headers={"Content-Type": "application/json"})
            res = urllib.request.urlopen(req)
            res = res.read().decode("utf-8")
            res = json.loads(res)
            print("stage status_code: ", results, res)
        except Exception as e:
            # 打印异常信息
            print('error: ', e)


@app.route('/cancelDiagnosis', methods=["POST"])
def cancelDiagnosis():
    try:
        data = request.get_json()
        file_id = data.get('sampleId')
        print("file_id: ", file_id)
        stop_event.set()
        return jsonify({'content': 200}), 200
    except Exception as e:
        print(f"stop error {e}")
        return jsonify({'content': 500}), 500


@app.route('/receive_svs_results', methods=['POST'])
def receive_svs_results():
    try:
        data = request.get_json()
        file_id = data.get('sampleId')
        print("receive_svs_results file_id: ", file_id)
        if not svs_output_queue_15.empty():
            results = svs_output_queue_15.get()
            # print("receive_svs_results results: ", results)
            return jsonify(results[file_id]), 200
        return jsonify({'error': 'An error occurred', 'message': 'The results have been taken out'}), 500
    except Exception as e:
        print('error: ', e)
        return jsonify({'error': 'An error occurred', 'message': str(e)}), 500


@app.route('/saveAllSamll', methods=['POST'])
def saveAllSamll():
    try:
        data = request.get_json()
        file_id = data.get('sampleId')
        print("saveAllSamll file_id: ", file_id)
        if not svs_output_queue_all.empty():
            results = svs_output_queue_all.get()
            # print("saveAllSamll results: ", results)
            return jsonify(results[file_id]), 200
        return jsonify({'error': 'An error occurred', 'message': 'The results have been taken out'}), 500
    except Exception as e:
        # 可以在这里记录到日志文件或其他地方
        return jsonify({'error': 'An error occurred', 'message': str(e)}), 500


@app.route('/sendHeath', methods=['POST'])
def sendHeath():
    return jsonify({'content': 200}), 200


if __name__ == "__main__":
    app.run(root_url, port=root_port, debug=True)
