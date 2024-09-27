import sys
sys.path.append("/home/bavon/project/SLFCD/SLFCD/extras/")
sys.path.append("/home/bavon/project/SLFCD/SLFCD/project/")
sys.path.append("/home/bavon/project/SLFCD/SLFCD/")

import os
import argparse
import yaml
import stat
import shutil
import json
import cv2
import h5py
from tqdm import tqdm
from shapely.geometry import box
import warnings
warnings.filterwarnings("ignore")

from clam.models.model_clam2 import Classifier
from clam.utils.file_utils import save_hdf5
from wsi_core.InferSVSProcess import WholeSlideImage
from clam.datasets.dataset_h5 import Whole_Slide_Bag_FP_all
from custom.model.cbam_ext import ResidualNet
from BoxDeduplication import merge_overlapping_rectangles, get_vertices
import pytorch_lightning as pl

# import torch.multiprocessing as mp
# mp.set_start_method("spawn", force=True)
from multiprocessing import Queue
import threading
from threading import Thread

import requests
import urllib.request
from flask import Flask, request, jsonify

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

import numpy as np
import random
np.random.seed(0)
random.seed(0)


parser = argparse.ArgumentParser(description='inference script')
parser.add_argument('--data_dir', type=str, default=r'D:\project\SLFCD\dataset\ais\data\1-CG23_18831_01.svs',
                    help='svs file')
parser.add_argument('--save_path', type=str, default="heatmaps/output")
parser.add_argument('--root_url', type=str, default="192.168.0.98")
parser.add_argument('--root_port', type=int, default=8088)
parser.add_argument('--receive_port', type=str, default=8091)
parser.add_argument('--config_file', type=str, default="infer.yaml")
parser.add_argument('--device', type=str, default="cuda:1")
args = parser.parse_args()
root_url, root_port, receive_port = args.root_url, args.root_port, args.receive_port

svs_input_queue = Queue(6)
svs_output_queue_15 = Queue(6)
stage_send_queue_multi = Queue(6)
svs_output_queue_all = Queue(6)
stop_event = threading.Event()
dzi_event = threading.Event()

stage_flag = {'ais': 1, 'hsil': 2, 'lsil': 3, 1: 'ais模型推理完成', 2: "hsil模型推理完成", 3: "lsil模型推理完成", 4: '样本模糊无效'}

infer_instance = None


app = Flask(__name__)
    
class CoolSystem(pl.LightningModule):
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()
        self.params = hparams
        self.model = ResidualNet("ImageNet", 50, 2, "cbam", image_size=224)

    def forward(self, x):
        x = self.model(x)
        return x


class Infer:
    def __init__(self, args):
        config_path = os.path.join('heatmaps/configs', args.config_file)
        config_dict = yaml.safe_load(open(config_path, 'r'))
        config_dict["data_arguments"]["data_dir"] = args.data_dir
        config_dict["data_arguments"]["save_path"] = args.save_path
        config_dict["model_arguments"]["device"] = args.device

        self.data_args = argparse.Namespace(**config_dict['data_arguments'])
        self.model_args = argparse.Namespace(**config_dict['model_arguments'])
        self.heatmap_args = argparse.Namespace(**config_dict['heatmap_arguments'])
        self.sample_args = argparse.Namespace(**config_dict['sample_arguments'])

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
        self.queues_in = Queue()
        self.queues_out = Queue()

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
        cls_model_1.eval()
        print('load cls ckpt path 1: {}'.format(self.model_args.cls_ckpt_path_1))

        # cls_model
        cls_model_2 = Classifier(2048, 4)
        ckpt = torch.load(self.model_args.cls_ckpt_path_2, map_location=self.model_args.device)
        cls_model_2.load_state_dict(ckpt, strict=False)
        cls_model_2.eval()
        cls_model_2 = cls_model_2.to(args.device)
        print('load cls ckpt path 2: {}'.format(self.model_args.cls_ckpt_path_2))

        # ais
        model_ais = CoolSystem.load_from_checkpoint(self.model_args.ais_ckpt_path)
        model_ais.to(self.model_args.device)
        model_ais.eval()
        print('load ais ckpt path: {}'.format(self.model_args.ais_ckpt_path))

        # hsil
        model_hsil = CoolSystem.load_from_checkpoint(self.model_args.hsil_ckpt_path)
        model_hsil.to(self.model_args.device)
        model_hsil.eval()
        print('load hsil ckpt path: {}'.format(self.model_args.hsil_ckpt_path))

        # lsil
        model_lsil = CoolSystem.load_from_checkpoint(self.model_args.lsil_ckpt_path)
        model_lsil.to(self.model_args.device)
        model_lsil.eval()
        print('first lsil ckpt path: {}'.format(self.model_args.lsil_ckpt_path))

        return cls_model_1, cls_model_2, model_ais, model_hsil, model_lsil

    def process_svs(self, svs_content, seg_params, patch_level, patch_size, step_size, name=None, flag=False):
        save_path, svs_path, svs_id = svs_content[0], svs_content[1], svs_content[2]
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
            return svs_id, WSI_object, save_path_hdf5, pt_save_path, h5_save_path
        else:
            return svs_id, WSI_object, save_path_hdf5, full_img, infer_box_img, small_img_15, small_img_All

    def viz_infer_show(self, name, svs_id, probs, coords, conf, patch_size, total_img_copy1, total_img_copy2,
                       result_path_small, result_path_all, result_path, stop_event):
        save_flag = False
        index = np.where(np.array(probs) > conf)[0]
        probs = np.array(probs)[index]
        coords = np.array(coords)[index]

        indices = np.argsort(-probs)
        coords = coords[indices].tolist()
        probs = probs[indices].tolist()

        new_coords, new_probs = [], []
        for coord, prob in zip(coords, probs):
            if stop_event.is_set():
                save_flag = True
                break
            if coord not in new_coords:
                if len(new_probs) < 200:
                    new_coords.append(coord)
                    new_probs.append(prob)
                else:
                    break
        
        if save_flag:
            return None, None
        
        coord_15, coord_all = [], []
        for coord, prob in tqdm(zip(new_coords, new_probs), total=len(new_coords), desc=f"save img {name} {svs_id}"):
            if stop_event.is_set():
                save_flag = True
                break
            region = [int(coord[0]), int(coord[1]), int(coord[0]) + patch_size, int(coord[1]) + patch_size]
            cv2.rectangle(total_img_copy2, (region[0], region[1]), (region[2], region[3]), (255, 0, 0), 10, 5)

            img_ori = total_img_copy1[region[1]:region[3], region[0]:region[2], :]
            img = cv2.resize(img_ori, (patch_size, patch_size))
            if len(os.listdir(result_path_small)) < 15:
                cv2.imwrite(f"{result_path_small}/[{region[0]},{region[1]},{region[2]},{region[3]}].jpg", img)
                coord_15.append(region)

            cv2.imwrite(f"{result_path_all}/[{region[0]},{region[1]},{region[2]},{region[3]}].jpg", img)
            coord_all.append(region)
        
        if save_flag:
            return None, None
        cv2.imwrite(result_path, total_img_copy2)

        return coord_15, coord_all

    def cls_svs(self, file_path, WSI_object, patch_level, patch_size, step_size, pt_save_path, h5_save_path, stop_event):
        cls_flag = False
        dataset = Whole_Slide_Bag_FP_all(file_path=file_path, WSI_object=WSI_object,
                                         patch_level=patch_level, patch_size=patch_size, slide_size=step_size,
                                         transforms=self.custom_transforms)

        loader = DataLoader(dataset=dataset, batch_size=self.model_args.batch_size, pin_memory=True, num_workers=self.model_args.num_workers)

        mode = 'w'
        with torch.no_grad():
            for batch, coords in tqdm(loader, total=len(loader)):
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

    def infer(self, name, svs_id, file_path, WSI_object, patch_level, patch_size, step_size, model, stop_event,
              stage_send_queue_multi, full_img, small_img_15, small_img_All, box_img_save):
        infer_flag = False
        dataset = Whole_Slide_Bag_FP_all(file_path=file_path, WSI_object=WSI_object,
                                         patch_level=patch_level, patch_size=patch_size, slide_size=step_size,
                                         transforms=self.custom_transforms)
        loader = DataLoader(dataset=dataset, batch_size=self.model_args.batch_size,
                            num_workers=self.model_args.num_workers, pin_memory=True)

        results = {'probs': [], 'coords': []}
        with torch.no_grad():
            for batch_img, batch_coord in tqdm(loader, total=len(loader), desc=f"infer svs {name} {svs_id}"):
                if stop_event.is_set():
                    infer_flag = True
                    break
                batch_img = batch_img.to(self.model_args.device)

                _, cls_emb = model(batch_img)
                output = torch.squeeze(cls_emb, dim=-1)
                probs = F.softmax(output, dim=-1)
                predicts = torch.max(probs, dim=-1)

                probs = predicts.values.cpu()
                label = predicts.indices.cpu()

                index = torch.where(label == 1)[0]
                probs_1 = probs[index].tolist()
                coord_1 = batch_coord[index].tolist()

                results["probs"].extend(probs_1)
                results["coords"].extend(coord_1)
        
        if infer_flag:
            return None
        image = dataset.image
        cv2.imwrite(full_img, image)

        if results["probs"]:
            total_img_copy1 = image.copy()
            total_img_copy2 = image.copy()
            sorted_pairs = sorted(zip(results["probs"], results["coords"]), key=lambda x: x[0], reverse=True)
            sorted_list1, sorted_list2 = zip(*sorted_pairs)

            coord_15, coord_all = self.viz_infer_show(name=name, svs_id=svs_id, probs=sorted_list1, coords=sorted_list2,
                                                      conf=self.model_args.conf, patch_size=patch_size,
                                                      total_img_copy1=total_img_copy1, total_img_copy2=total_img_copy2,
                                                      result_path_small=small_img_15, result_path_all=small_img_All,
                                                      result_path=box_img_save, stop_event=stop_event)

            if coord_15 is None:
                return None
            
            if coord_all:
                rectangles = [box(*i) for i in coord_all]
                merged_rectangles = merge_overlapping_rectangles(rectangles)
                box_point_vertices = get_vertices(merged_rectangles)
            else:
                box_point_vertices = []
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

    def thread_all(self, stage_send_queue_multi, q_in, q_out, stop_event):
        # ----------------------------- ais_hsil ------------------------------
        self.patch_level_ais_hsil = self.ais_params.patch_level
        self.patch_size_ais_hsil = self.ais_params.patch_size
        self.slide_size_ais_hsil = self.ais_params.slide_size

        # ----------------------------- lsil ------------------------------
        self.patch_level_lsil = self.lsil_params.patch_level
        self.patch_size_lsil = self.lsil_params.patch_size
        self.slide_size_lsil = self.lsil_params.slide_size
        
        while True:
            while not stop_event.is_set():
                results = {}

                svs_content = q_in.get()
                # ------------------ 对 SVS 进行分类检测的前处理 ------------------
                if stop_event.is_set():
                    break
                seg_params = self.def_seg_params.copy()
                svs_id, WSI_object, save_path_hdf5, \
                    pt_save_path, h5_save_path = self.process_svs(svs_content, seg_params,
                                                                  self.model_args.cls_arguments["patch_level"],
                                                                  self.model_args.cls_arguments["patch_size"],
                                                                  self.model_args.cls_arguments["step_size"],
                                                                  flag=True)

                # ------------------ 对 SVS 进行分类检测的推理处理 ------------------
                if stop_event.is_set():
                    break
                if os.path.exists(save_path_hdf5):
                    infer_category = self.cls_svs(save_path_hdf5, WSI_object,
                                                  self.model_args.cls_arguments["patch_level"],
                                                  self.model_args.cls_arguments["patch_size"],
                                                  self.model_args.cls_arguments["step_size"],
                                                  pt_save_path, h5_save_path, stop_event)

                    self.results[svs_id]['category'] = infer_category
                    
                    if stop_event.is_set():
                        break
                        
                    if infer_category == 'ais':
                        # ----------------------------- ais infer ------------------------------
                        seg_params = self.def_seg_params.copy()
                        svs_id, WSI_object, save_path_hdf5, full_img, infer_box_img, \
                            small_img_15, small_img_All = self.process_svs(svs_content, seg_params,
                                                                           self.patch_level_ais_hsil,
                                                                           self.patch_size_ais_hsil,
                                                                           self.slide_size_ais_hsil,
                                                                           name='ais')

                        if stop_event.is_set():
                            break
                        result_ais = self.infer('ais', svs_id, save_path_hdf5, WSI_object,
                                                self.patch_level_ais_hsil,
                                                self.patch_size_ais_hsil,
                                                self.slide_size_ais_hsil,
                                                self.model_ais, stop_event, stage_send_queue_multi,
                                                full_img, small_img_15, small_img_All, infer_box_img)
                        if stop_event.is_set():
                            break
                        
                        results.update(result_ais)

                        # ----------------------------- hsil infer ------------------------------
                        seg_params = self.def_seg_params.copy()
                        svs_id, WSI_object, save_path_hdf5, full_img, infer_box_img, \
                            small_img_15, small_img_All = self.process_svs(svs_content, seg_params,
                                                                           self.patch_level_ais_hsil,
                                                                           self.patch_size_ais_hsil,
                                                                           self.slide_size_ais_hsil,
                                                                           name='hsil')

                        if stop_event.is_set():
                            break

                        result_hsil = self.infer('hsil', svs_id, save_path_hdf5, WSI_object,
                                                 self.patch_level_ais_hsil,
                                                 self.patch_size_ais_hsil,
                                                 self.slide_size_ais_hsil,
                                                 self.model_hsil, stop_event, stage_send_queue_multi,
                                                 full_img, small_img_15, small_img_All, infer_box_img)
                        if stop_event.is_set():
                            break
                        results.update(result_hsil)

                        # ----------------------------- lsil infer ------------------------------
                        seg_params = self.def_seg_params.copy()
                        svs_id, WSI_object, save_path_hdf5, full_img, infer_box_img, \
                            small_img_15, small_img_All = self.process_svs(svs_content, seg_params,
                                                                           self.patch_level_lsil,
                                                                           self.patch_size_lsil,
                                                                           self.slide_size_lsil,
                                                                           name='lsil')

                        if stop_event.is_set():
                            break

                        result_lsil = self.infer('lsil', svs_id, save_path_hdf5, WSI_object,
                                                 self.patch_level_lsil,
                                                 self.patch_size_lsil,
                                                 self.slide_size_lsil,
                                                 self.model_lsil, stop_event,stage_send_queue_multi, 
                                                 full_img, small_img_15, small_img_All, infer_box_img)
                        if stop_event.is_set():
                            break
                        results.update(result_lsil)

                    elif infer_category == 'hsil':
                        # ----------------------------- hsil infer ------------------------------
                        seg_params = self.def_seg_params.copy()
                        svs_id, WSI_object, save_path_hdf5, full_img, infer_box_img, \
                            small_img_15, small_img_All = self.process_svs(svs_content, seg_params,
                                                                           self.patch_level_ais_hsil,
                                                                           self.patch_size_ais_hsil,
                                                                           self.slide_size_ais_hsil,
                                                                           name='hsil')

                        if stop_event.is_set():
                            break

                        result_hsil = self.infer('hsil', svs_id, save_path_hdf5, WSI_object,
                                                 self.patch_level_ais_hsil,
                                                 self.patch_size_ais_hsil,
                                                 self.slide_size_ais_hsil,
                                                 self.model_hsil, stop_event, stage_send_queue_multi, 
                                                 full_img, small_img_15, small_img_All, infer_box_img)
                        if stop_event.is_set():
                            break
                        results.update(result_hsil)

                        # ----------------------------- lsil infer ------------------------------
                        seg_params = self.def_seg_params.copy()
                        svs_id, WSI_object, save_path_hdf5, full_img, infer_box_img, \
                            small_img_15, small_img_All = self.process_svs(svs_content, seg_params,
                                                                           self.patch_level_lsil,
                                                                           self.patch_size_lsil,
                                                                           self.slide_size_lsil,
                                                                           name='lsil')

                        if stop_event.is_set():
                            break

                        result_lsil = self.infer('lsil', svs_id, save_path_hdf5, WSI_object,
                                                 self.patch_level_lsil,
                                                 self.patch_size_lsil,
                                                 self.slide_size_lsil,
                                                 self.model_lsil, stop_event, stage_send_queue_multi, 
                                                 full_img, small_img_15, small_img_All, infer_box_img)
                        if stop_event.is_set():
                            break
                        results.update(result_lsil)

                    elif infer_category == 'lsil':
                        # ----------------------------- lsil infer ------------------------------
                        seg_params = self.def_seg_params.copy()
                        svs_id, WSI_object, save_path_hdf5, full_img, infer_box_img, \
                            small_img_15, small_img_All = self.process_svs(svs_content, seg_params, 
                                                                           self.patch_level_lsil,
                                                                           self.patch_size_lsil,
                                                                           self.slide_size_lsil,
                                                                           name='lsil')

                        if stop_event.is_set():
                            break

                        result_lsil = self.infer('lsil', svs_id, save_path_hdf5, WSI_object,
                                                 self.patch_level_lsil,
                                                 self.patch_size_lsil,
                                                 self.slide_size_lsil,
                                                 self.model_lsil, stop_event, stage_send_queue_multi,
                                                 full_img, small_img_15, small_img_All, infer_box_img)
                        if stop_event.is_set():
                            break
                        results.update(result_lsil)

                    elif infer_category == 'normal':
                        results = {'ais': {'coord_15': [], 'coord_all': [], 'size': (9134, 7666), 'zoom': (1, 4), 'level': 1, 'box_point_vertices': []},
                                   'hsil': {'coord_15': [], 'coord_all': [], 'size': (9134, 7666), 'zoom': (1, 4), 'level': 1, 'box_point_vertices': []},
                                   'lsil': {'coord_15': [], 'coord_all': [], 'size': (9134, 7666), 'zoom': (1, 4), 'level': 1, 'box_point_vertices': []}}
                    
                    stop_event.set()
                    q_out.put(results)
                else:
                    q_out.put([[]])
            
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
            self.results_0[file_id]['level']['ais'] = 1
            self.results_0[file_id]['level']['hsil'] = 1
            self.results_0[file_id]['level']['lsil'] = 1
            self.results_0[file_id]['box_point_vertices']['ais'] = []
            self.results_0[file_id]['box_point_vertices']['hsil'] = []
            self.results_0[file_id]['box_point_vertices']['lsil'] = []

    def main(self, save_url, file_path, file_id, stage_send_queue_multi, svs_output_queue_15, svs_output_queue_all, stop_event, dzi_event):
        self.results["infer_id"] = file_id
        self.results[file_id] = self.orig_results

        self.results_0["infer_id"] = file_id
        self.results_0[file_id] = self.orig_results_0

        svs_thread = threading.Thread(target=self.thread_all, args=(stage_send_queue_multi, self.queues_in, self.queues_out, stop_event))
        svs_thread.start()

        self.queues_in.put([save_url, file_path, file_id])
        results = self.queues_out.get()

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


def get_infer_instance(args):
    global infer_instance
    if infer_instance is None:
        infer_instance = Infer(args)
    return infer_instance


@app.route('/upload_svs_file', methods=['POST'])
def upload_svs_file():
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
        process_infer_thread = Thread(target=process_svs_file, args=(args, save_url, file_url, file_id,
                                                                     stage_send_queue_multi, svs_output_queue_15,
                                                                     svs_output_queue_all, stop_event, dzi_event))
        process_infer_thread.start()

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


def process_svs_file(args, save_url, file_path, file_id, stage_send_queue_multi, svs_output_queue_15, svs_output_queue_all, stop_event, dzi_event):
    if not stage_send_queue_multi.empty():
        stage_send_queue_multi.get()
    print("stage 0 文件正在分析")
    infer = get_infer_instance(args) 
    stage_send_queue_multi.put({'sampleId': file_id, 'state': 0})
    infer.main(save_url, file_path, file_id, stage_send_queue_multi, svs_output_queue_15, svs_output_queue_all, stop_event, dzi_event)


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
            print("receive_svs_results results: ", results)
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
            print("saveAllSamll results: ", results)
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
    
    
