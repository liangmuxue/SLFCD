import sys
import os
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
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from clam.datasets.dataset_h5 import Dataset_All_Bags
from clam.datasets.dataset_combine import Whole_Slide_Bag_COMBINE
from clam.utils.utils import print_network, collate_features

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from camelyon16.data.image_producer import ImageDataset


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

def chose_model(model_name):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
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
        model.fc = nn.Linear(fc_features, 1)        
        self.model = model.to(device)
        self.loss_fn = BCEWithLogitsLoss().to(device)
        self.loss_fn.requires_grad_(True)
        self.save_hyperparameters()
        
    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        
        optimizer = torch.optim.SGD([
                {'params': self.model.parameters()},
            ], lr=self.params.lr, momentum=self.params.momentum)

        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        return [optimizer], [exp_lr_scheduler]

    def training_step(self, batch, batch_idx):
        """training"""
        
        x, y = batch
        y = y.float()
        output = self.model.forward(x)
        output = torch.squeeze(output,dim=-1) 
        loss = self.loss_fn(output, y)
        probs = output.sigmoid()
        predicts = (probs >= 0.5).type(torch.float)

        acc = (predicts == y).sum().data * 1.0 / self.params.batch_size

        self.log('train_loss', loss, batch_size=batch[0].shape[0], prog_bar=True)
        self.log('train_acc', acc, batch_size=batch[0].shape[0], prog_bar=True)
        
        return {'loss': loss, 'train_acc': acc}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y = y.float()
        output = self.model.forward(x)
        output = torch.squeeze(output,dim=-1) 
        loss = self.loss_fn(output, y)
        probs = output.sigmoid()
        predicts = (probs >= 0.5).type(torch.float)
        acc = (predicts == y).type(torch.float).sum().data * 1.0 / self.params.batch_size
        
        self.log('val_loss', loss, batch_size=batch[0].shape[0], prog_bar=True)
        self.log('val_acc', acc, batch_size=batch[0].shape[0], prog_bar=True)

        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        _, preds = torch.max(y_hat, 1)
        acc = torch.sum(preds == y.data) / (y.shape[0] * 1.0)

        return {'test_loss': loss, 'test_acc': acc}

    def train_dataloader(self):
        hparams = self.params
        file_path = hparams.data_path
        csv_path = os.path.join(file_path,"train.csv")
        split_data = pd.read_csv(csv_path).values[:,0].tolist()
        wsi_path = os.path.join(file_path,"data")
        mask_path = os.path.join(file_path,"tumor_mask")
        dataset_train = Whole_Slide_Bag_COMBINE(file_path,wsi_path,mask_path,split_data=split_data,patch_level=hparams.patch_level)
    
        train_loader = DataLoader(dataset_train,
                                      batch_size=self.params.batch_size,
                                      num_workers=self.params.num_workers)
        return train_loader

    def val_dataloader(self):
        hparams = self.params
        file_path = hparams.data_path
        csv_path = os.path.join(file_path,"valid.csv")
        split_data = pd.read_csv(csv_path).values[:,0].tolist()
        wsi_path = os.path.join(file_path,"data")
        mask_path = os.path.join(file_path,"tumor_mask")
        dataset_valid = Whole_Slide_Bag_COMBINE(file_path,wsi_path,mask_path,split_data=split_data)
        val_loader = DataLoader(dataset_valid,
                                      batch_size=self.params.batch_size,
                                      num_workers=self.params.num_workers)
        return val_loader

    # def test_dataloader(self):
    #     transform = transforms.Compose([
    #                           transforms.Resize(256),
    #                           transforms.CenterCrop(224),
    #                           transforms.ToTensor(),
    #                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #                           ])
    #
    #     val_set = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform)
    #     val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=True, num_workers=4)
    #     return val_loader


def get_last_ck_file(checkpoint_path):
    list = os.listdir(checkpoint_path)
    list.sort(key=lambda fn: os.path.getmtime(checkpoint_path+"/"+fn) if not os.path.isdir(checkpoint_path+"/"+fn) else 0)    
    return list[-1]

def main(hparams):
    checkpoint_path = hparams.work_dir + "/checkpoints"
    filename = 'slfcd-{epoch:02d}-{val_loss:.2f}'
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_path,
        filename=filename,
        save_top_k=3,
        auto_insert_metric_name=False
    ) 
    model_logger = (
        pl_loggers.TensorBoardLogger(save_dir=hparams.work_dir, name="resnet18_log", version="logs")
    )              
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
        model = CoolSystem(hparams)
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            gpus=1,
            accelerator='gpu',
            logger=model_logger,
            callbacks=[checkpoint_callback],
            log_every_n_steps=1
        )  
        trainer.fit(model)
        


if __name__ == '__main__':
    cnn_path = 'custom/configs/config.json'
    with open(cnn_path, 'r') as f:
        args = json.load(f) 
    hyperparams = Namespace(**args)    
    main(hyperparams)
    
    
