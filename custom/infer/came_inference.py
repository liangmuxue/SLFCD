import sys
import os
import shutil
import argparse
import logging
import json
import time
from argparse import Namespace
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.optim import SGD
from torchvision import models
from torch import nn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import cv2

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.states import RunningStage
import numpy as np
import pickle

from clam.datasets.dataset_h5 import Dataset_All_Bags
from clam.datasets.dataset_combine import Whole_Slide_Bag_COMBINE,Whole_Slide_Det
from clam.datasets.dataset_inference import Whole_Slide_Bag_Infer
from clam.utils.utils import print_network, collate_features
from camelyon16.data.image_producer import ImageDataset
from utils.constance import get_label_cate,get_label_cate_num
from utils.wsi_img_viz import viz_infer_dataset
from custom.model.cbam_ext import ResidualNet
from custom.train_with_clamdata import CoolSystem,get_last_ck_file
from utils.vis import visdom_data
from visdom import Visdom

device = torch.device('cuda:0')
viz_tumor = Visdom(env="tumor_viz", port=8098)

def main(hparams,device_ids=None,single_name=None,result_path=None,slide_size=128):
    checkpoint_path = os.path.join(hparams.work_dir,"checkpoints",hparams.model_name)
    filename = 'slfcd-{epoch:02d}-{val_loss:.2f}'
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_path,
        filename=filename,
        save_top_k=3,
        auto_insert_metric_name=False
    ) 
    logger_name = "app_log"
    model_logger = (
        pl_loggers.TensorBoardLogger(save_dir=hparams.work_dir, name=logger_name, version=hparams.model_name)
    )             
    log_path = os.path.join(hparams.work_dir,logger_name,hparams.model_name) 
    
    if hparams.load_weight:
        file_name = get_last_ck_file(checkpoint_path)
        checkpoint_path_file = "{}/{}".format(checkpoint_path,file_name)
        # model = torch.load(checkpoint_path_file) # 
        model = CoolSystemInfer.load_from_checkpoint(checkpoint_path_file).to(device)
        # 使用当前配置里的超参数
        model.params = hparams
        model.result_path = result_path
        model.file_name = single_name

    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        gpus=[int(device_ids)],
        accelerator='gpu',
        logger=model_logger,
        log_every_n_steps=1
    )         
    trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(224),
            ])  
            
    dataset_infer = Whole_Slide_Bag_Infer(hparams.data_path,single_name,patch_size=hparams.image_size,
                                slide_size=slide_size,patch_level=hparams.patch_level,transform=trans)    
    inference_loader = DataLoader(dataset_infer,
                                  batch_size=64,
                                  collate_fn=model._collate_fn,
                                  shuffle=False,
                                  num_workers=1)        
    print("total len:{}".format(len(dataset_infer)))     
    predictions = trainer.predict(model=model,dataloaders=inference_loader)

def viz_results(hparams,single_name=None,result_path=None,slide_size=128):
    """可视化"""
    dataset_infer = Whole_Slide_Bag_Infer(hparams.data_path,single_name,patch_size=hparams.image_size,
                                slide_size=slide_size,patch_level=hparams.patch_level,transform=None)  
    
    save_path = os.path.join(result_path,"ret_{}.pkl".format(single_name))
    loader = open(save_path,'rb')
    result_data = pickle.load(loader)
    viz_infer_dataset(result_data,dataset=dataset_infer,result_path=result_path) 

def single_img_inference(img_path,hparams=None):
    checkpoint_path = os.path.join(hparams.work_dir,"checkpoints",hparams.model_name)
    file_name = get_last_ck_file(checkpoint_path)
    checkpoint_path_file = "{}/{}".format(checkpoint_path,file_name)
    model = CoolSystemInfer.load_from_checkpoint(checkpoint_path_file).to(device)
    # 使用当前配置里的超参数
    model.params = hparams
    model.result_path = result_path 
    img = cv2.imread(img_path)
    model.single_inference(img)
          
class CoolSystemInfer(CoolSystem):
    
    def __init__(self, hparams,source=None,file_name=None,device=None):
        super(CoolSystemInfer, self).__init__(hparams,device=device)    
        self.file_name = file_name
        self.results = []
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """根据给出的数据文件，判断patch区域所属类别"""
        
        img_tar, coord, item,img = batch
        batch_size = img_tar.shape[0]
        results = np.zeros((batch_size,2))
        
        img_tar = img_tar.to(device)
        feature,cls_out = self.model(img_tar)
        probs = F.softmax(cls_out,dim=-1)
        predicts = torch.max(probs,dim=-1)[1]   
        tum_idx = torch.where(predicts==1)[0]
        if tum_idx.shape[0]>0:
            probs = probs.cpu().numpy()
            tum_idx = tum_idx.cpu().numpy()
            predicts = predicts.cpu().numpy()
            print("has pred 1:{}".format(tum_idx))
            img_ori = np.array(img)[tum_idx]
            for j in range(img_ori.shape[0]):
                win = "win_{}".format(tum_idx[j])
                probs_show = int(round(probs[tum_idx[j]][1],2)*100)
                title = "{}_{}".format(probs_show,coord[tum_idx[j]].cpu().numpy().tolist())
                visdom_data(img_ori[j],[], win=win,title=title,viz=viz_tumor)
                item[tum_idx[j]]["pred"] = 1
                item[tum_idx[j]]["probs"] = probs_show
            results[tum_idx,0] = 1
            results[tum_idx,1] = probs[tum_idx,1]
        
        self.results.append(item)       
        
    def on_predict_end(self):
        results = np.concatenate(np.array(self.results))
        save_path = os.path.join(self.result_path,"ret_{}.pkl".format(self.file_name))
        writer = open(save_path,'wb')
        pickle.dump(results, writer)
        writer.close()                
        # np.save(save_path,self.results)

    def single_inference(self,img):
        img = cv2.resize(img,(224,224))
        img_tar = torch.Tensor(img).to(device).permute(2,0,1).unsqueeze(0)
        feature,cls_out = self.model(img_tar)
        probs = F.softmax(cls_out,dim=-1)
        predicts = torch.max(probs,dim=-1)[1]  
        print("predicts:{},probs:{}".format(predicts,probs))        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--device_ids', default='0', type=str, help='choose device')
    parser.add_argument('--mode', default='lsil', type=str, help='choose type')
    parser.add_argument('--result_path', default='', type=str)
    parser.add_argument('--slide_size', default=128, type=int)
    parser.add_argument('--inf_filename', default='2-CG23_10410_02', type=str)
    args = parser.parse_args()
    device_ids = args.device_ids
    result_path = args.result_path
    slide_size = args.slide_size
    single_name = args.inf_filename
    single_name = "6-CG23_12974_06"
    # cnn_path = 'custom/configs/config_lsil.json'
    # cnn_path = 'custom/configs/config_hsil.json'
    if args.mode=="hsil":
        cnn_path = 'custom/configs/config_hsil_liang.json'
    if args.mode=="lsil":
        cnn_path = 'custom/configs/config_lsil_liang.json'        
    with open(cnn_path, 'r') as f:
        args = json.load(f) 
    
    
    hyperparams = Namespace(**args)   
    # single_name = "80-CG23_15274_01"
    # single_name = "2-CG23_10410_02"
    # main(hyperparams,device_ids=device_ids,single_name=single_name,result_path=result_path,slide_size=slide_size)
    viz_results(hyperparams,single_name=single_name,result_path=result_path,slide_size=slide_size)
    img_path = "results/infer_prop/input/test1.png"
    # single_img_inference(img_path,hparams=hyperparams)
            
        
        