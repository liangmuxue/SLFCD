import sys
import os
import shutil
import argparse
import logging
import json
import time
from argparse import Namespace
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.optim import SGD
from torchvision import models
from torch import nn
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
# import torch.nn.LSTM
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.states import RunningStage
import numpy as np
from clam.datasets.dataset_h5 import Dataset_All_Bags
# from clam.datasets.dataset_combine import Whole_Slide_Bag_COMBINE
from clam.datasets.dataset_combine_together import  Whole_Slide_Bag_COMBINE_togeter
from clam.utils.utils import print_network, collate_features
import types

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from camelyon16.data.image_producer import ImageDataset
from utils.constance import get_label_cate

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cnn_path', default=None, metavar='CNN_PATH', type=str,
                    help='Path to the config file in json format')
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help='Path to the saved models')
parser.add_argument('--num_workers', default=2, type=int, help='number of'
                    ' workers for each data loader, default 2.')
parser.add_argument('--device_ids', default='0', type=str, help='comma'
                    ' separated indices of GPU to use, e.g. 0,1 for using GPU_0'
                    ' and GPU_1, default 0.')

device = 'cuda:0' # torch.device('cuda:0')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# device = torch.device('cpu')

from utils.vis import vis_data,visdom_data
from visdom import Visdom

viz_tumor_train = Visdom(env="tumor_train", port=8098)
viz_tumor_valid = Visdom(env="tumor_valid", port=8098)
viz_normal_train = Visdom(env="normal_train", port=8098)
viz_normal_valid = Visdom(env="normal_valid", port=8098)

def chose_model(model_name):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=False)         
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=False)        
    else:
        raise Exception("I have not add any models. ")
    return model



class CoolSystem(pl.LightningModule):


    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams
            
        ########## define the model ########## 
        model = chose_model(hparams.model)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, len(get_label_cate()))        
        self.model = model.to(device)
        self.loss_fn = nn.CrossEntropyLoss().to(device)
        self.loss_fn.requires_grad_(True)
        self.save_hyperparameters()
        
        self.resuts = None
        
    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        
        optimizer = torch.optim.SGD([
                {'params': self.model.parameters()},
            ], lr=self.params.lr, momentum=self.params.momentum)
        optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},
            ],  weight_decay=1e-4,lr=self.params.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,gamma=0.3, step_size=5)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer,base_lr=1e-4,max_lr=1e-3,step_size_up=30)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=16,eta_min=1e-4)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """training"""
        
        x, y,img_ori,_ = batch
        output = self.model.forward(x)
        output = torch.squeeze(output,dim=-1) 
        loss = self.loss_fn(output, y)
        predicts = F.softmax(output,dim=-1)
        predicts = torch.max(predicts,dim=-1)[1] 

        acc = (predicts == y).sum().data * 1.0 / self.params.batch_size

        self.log('train_loss', loss, batch_size=batch[0].shape[0], prog_bar=True)
        self.log('train_acc', acc, batch_size=batch[0].shape[0], prog_bar=True)
        self.log("lr",self.trainer.optimizers[0].param_groups[0]["lr"], batch_size=batch[0].shape[0], prog_bar=True)
        
        # Sample Viz
        # tumor_index = torch.where(y>0)[0]
        # for index in tumor_index:
        #     if np.random.randint(1,10)==3:
        #         ran_idx = np.random.randint(1,10)
        #         win = "win_{}".format(ran_idx)
        #         label = y[index]
        #         sample_img = img_ori[index]
        #         title = "label{}_{}".format(label,ran_idx)
        #         visdom_data(sample_img, [], viz=viz_tumor_train,win=win,title=title) 
        # normal_index = torch.where(y==0)[0]    
        # for index in normal_index:
        #     if np.random.randint(1,50)==3:
        #         ran_idx = np.random.randint(1,10)
        #         win = "win_{}".format(ran_idx)
        #         label = y[index]
        #         sample_img = img_ori[index]
        #         title = "label{}_{}".format(label,ran_idx)
        #         visdom_data(sample_img, [], viz=viz_normal_train,win=win,title=title)                     
        return {'loss': loss, 'train_acc': acc}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y,img_ori,_ = batch
        output = self.model.forward(x)
        output = torch.squeeze(output,dim=-1) 
        loss = self.loss_fn(output, y)
        predicts = F.softmax(output,dim=-1)
        predicts = torch.max(predicts,dim=-1)[1]    
        pred_acc_bool = (predicts == y)
        acc = pred_acc_bool.type(torch.float).sum().data * 1.0 / self.params.batch_size
        
        # Calculate the accuracy of each category separately
        all_labes = get_label_cate()
        results = []
        for label in all_labes:
            pred_index = torch.where(predicts==label)[0]
            acc_cnt = torch.sum(y[pred_index]==label)
            fail_cnt = torch.sum(y[pred_index]!=label)
            label_cnt = torch.sum(y==label)
            results.append([label,acc_cnt.cpu().item(),fail_cnt.cpu().item(),label_cnt.cpu().item()])
            
        # Sample Viz
        tumor_index = torch.where(y>0)[0]
        
        for index in tumor_index:
            if np.random.randint(1,10)==3:
                ran_idx = np.random.randint(1,10)
                win = "win_{}".format(ran_idx)
                label = y[index]
                sample_img = img_ori[index]
                title = "label{}_{}".format(label,ran_idx)
                visdom_data(sample_img, [], viz=viz_tumor_valid,win=win,title=title) 
        normal_index = torch.where(y==0)[0]    
        for index in normal_index:
            if np.random.randint(1,50)==3:
                ran_idx = np.random.randint(1,10)
                win = "win_{}".format(ran_idx)
                label = y[index]
                sample_img = img_ori[index]
                title = "label{}_{}".format(label,ran_idx)
                visdom_data(sample_img, [], viz=viz_normal_valid,win=win,title=title)      
                     
        results = np.array(results)
                
        if self.results is None:
            self.results = results    
        else:
            self.results = np.concatenate((self.results,results),axis=0)
        
        self.log('val_loss', loss, batch_size=batch[0].shape[0], prog_bar=True)
        self.log('val_acc', acc, batch_size=batch[0].shape[0], prog_bar=True)

        return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_start(self):
        self.results = None
        
    def on_validation_epoch_end(self):
        # For summary calculation
        # if self.trainer.state.stage==RunningStage.SANITY_CHECKING:
        #     return   
        
        columns = ["label","acc_cnt","fail_cnt","real_cnt"]
        results_pd = pd.DataFrame(self.results,columns=columns)
        
        all_labes = get_label_cate()
        for label in all_labes:
            acc_cnt = results_pd[results_pd["label"]==label]["acc_cnt"].sum()
            fail_cnt = results_pd[results_pd["label"]==label]["fail_cnt"].sum()
            real_cnt = results_pd[results_pd["label"]==label]["real_cnt"].sum()
            self.log('acc_cnt_{}'.format(label), float(acc_cnt), prog_bar=True)
            self.log('fail_cnt_{}'.format(label), float(fail_cnt), prog_bar=True)
            self.log('real_cnt_{}'.format(label), float(real_cnt), prog_bar=True)
            if acc_cnt+fail_cnt==0:
                acc = 0.0
            else:
                acc = acc_cnt/(acc_cnt+fail_cnt)
            if real_cnt==0:
                recall = 0.0
            else:
                recall = acc_cnt/real_cntl
            self.log('acc_{}'.format(label), acc, prog_bar=True)
            self.log('recall_{}'.format(label), recall, prog_bar=True)
                     
    def train_dataloader(self):
        hparams = self.params
        # types = hparams.type
        # split_data_total = []
        # file_path = hparams.data_path                                  #  type
        # tumor_mask_path = hparams.tumor_mask_path
        # csv_path = os.path.join(file_path,hparams.train_csv)           #type
        # split_data = pd.read_csv(csv_path).values[:,0].tolist()        #type
            # if split_data_total is None:
            #     split_data_total = split_data
            # else:
            #     split_data_total = combine(split_data_total,split_data)
        # wsi_path = os.path.join(file_path,"data")                     #type
        # mask_path = os.path.join(file_path,tumor_mask_path)          #type
        # dataset_train = Whole_Slide_Bag_COMBINE(file_path,wsi_path,mask_path,work_type="train",patch_path=hparams.patch_path,
        #                                             patch_size=hparams.image_size,split_data=split_data,patch_level=hparams.patch_level)
        
        dataset_train = Whole_Slide_Bag_COMBINE_togeter(hparams,work_type="train",patch_size=hparams.image_size,patch_level=hparams.patch_level)
        train_loader = DataLoader(dataset_train,
                                          batch_size=self.params.batch_size,
                                          collate_fn=self._collate_fn,
                                          shuffle=True,
                                          num_workers=self.params.num_workers)
        # data_summarize(train_loader)
        return train_loader

    def val_dataloader(self):
        hparams = self.params
        # types  = hparams.type
        #
        # file_path = hparams.data_path
        # tumor_mask_path = hparams.tumor_mask_path
        # csv_path = os.path.join(file_path,hparams.valid_csv)
        # split_data = pd.read_csv(csv_path).values[:,0].tolist()
        # wsi_path = os.path.join(file_path,"data")
        # mask_path = os.path.join(file_path,tumor_mask_path)
        # dataset_valid = Whole_Slide_Bag_COMBINE(file_path,wsi_path,mask_path,work_type="valid",patch_path=hparams.patch_path,
        #                                                 patch_size=hparams.image_size,split_data=split_data,patch_level=hparams.patch_level,
        #                                                 )
        
        dataset_valid = Whole_Slide_Bag_COMBINE_togeter(hparams,work_type='valid',patch_size=hparams.image_size,patch_level=hparams.patch_level)
        val_loader = DataLoader(dataset_valid,
                                              batch_size=self.params.batch_size,
                                              collate_fn=self._collate_fn,
                                              shuffle=False,
                                              num_workers=self.params.num_workers)
            
        return val_loader

    
    def _collate_fn(self,batch):
        
        first_sample = batch[0]
        aggregated = []
        for i in range(len(first_sample)):
            if i==0:
                sample_list = [sample[i] for sample in batch]
                aggregated.append(
                    torch.stack(sample_list, dim=0)
                )
            elif i==1:
                sample_list = [sample[i] for sample in batch]
                aggregated.append(torch.from_numpy(np.array(sample_list)))
            else:
                aggregated.append([sample[i] for sample in batch])                
        return aggregated       
        

def get_last_ck_file(checkpoint_path):
    list = os.listdir(checkpoint_path)
    list.sort(key=lambda fn: os.path.getmtime(checkpoint_path+"/"+fn) if not os.path.isdir(checkpoint_path+"/"+fn) else 0)    
    return list[-1]

def main(hparams):
    checkpoint_path = os.path.join(hparams.work_dir,"checkpoints",hparams.model_name)
    print(checkpoint_path)
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
        model = CoolSystem.load_from_checkpoint(checkpoint_path_file).to(device)
        # trainer = Trainer(resume_from_checkpoint=checkpoint_path_file)
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            gpus=1,
            accelerator='gpu',
            logger=model_logger,
            callbacks=[checkpoint_callback],
            log_every_n_steps=1
        )       
        trainer.fit(model,ckpt_path=checkpoint_path_file)   
    else:
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        os.mkdir(checkpoint_path)
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        os.mkdir(log_path)
        
        model = CoolSystem(hparams)
        # data_summarize(model.val_dataloader())
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            gpus=1,
            accelerator='gpu',
            logger=model_logger,
            callbacks=[checkpoint_callback],
            log_every_n_steps=1
        )  
        trainer.fit(model)
        
def data_summarize(dataloader):
    
    it = iter(dataloader)
    size = len(dataloader)
    viz_number_tumor = 0
    viz_number_normal = 0        
    label_stat = []
    for index in range(size):
        img,label,img_ori,item = next(it)
        img_ori = img_ori.cpu().numpy()[0]
        type = item['type'][0]
        if type=="annotation":
            label_stat.append(item['label'])
            if viz_number_tumor<10:
                viz_number_tumor += 1
                visdom_data(img_ori,[],title="tumor_{}".format(index), viz=viz_tumor_valid)
        else:
            label_stat.append(0)
            if viz_number_normal<10:
                viz_number_normal += 1                
                visdom_data(img_ori,[], title="normal_{}".format(index),viz=viz_normal_valid)
    
    label_stat = np.array(label_stat)
    
    print("label_stat 1:{},2:{},3:{}".format(np.sum(label_stat==1),np.sum(label_stat==2),np.sum(label_stat==3)))

if __name__ == '__main__':
    cnn_path = 'custom/configs/config_together.json'
    with open(cnn_path, 'r') as f:
        args = json.load(f) 
    hyperparams = Namespace(**args)    
    main(hyperparams)
    
    
