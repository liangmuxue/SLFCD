import sys
sys.path.append('/home/bavon/project/SLFCD/SLFCD')
sys.path.append('/home/bavon/project/SLFCD/SLFCD/project')
import os
import argparse
import json
from argparse import Namespace
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn
import torchvision.transforms as transforms

import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

from clam.datasets.dataset_combine import Whole_Slide_Bag_COMBINE, Whole_Slide_Bag_COMBINE_all
from utils.constance import get_label_cate, get_label_cate_num
from custom.model.cbam_ext import ResidualNet
import cv2

import torch
torch.backends.cudnn.enabled = False

import warnings
warnings.filterwarnings('ignore')

from utils.vis import visdom_data
from visdom import Visdom

viz_tumor_train = Visdom(env="tumor_train")
viz_tumor_valid = Visdom(env="tumor_valid")
viz_normal_train = Visdom(env="normal_train")
viz_normal_valid = Visdom(env="normal_valid")

device = 'cuda:1'


def chose_model(model_name, num_classes, image_size=512):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'cbam':
        model = ResidualNet("ImageNet", 50, num_classes, "cbam", image_size=image_size)
    else:
        raise Exception("I have not add any models. ")
    return model


class CoolSystem(pl.LightningModule):
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()
        self.params = hparams

        # define the model
        self.model = chose_model(hparams.model, len(get_label_cate(mode=hparams.mode)), image_size=hparams.image_size)
        self.model = self.model.to(device)
        self.loss_fn = nn.CrossEntropyLoss().to(device)
        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD([{'params': self.model.parameters()}],
        #                             lr=self.params.lr, momentum=self.params.momentum)

        optimizer = torch.optim.Adam([{'params': self.model.parameters()}],
                                     weight_decay=1e-4, lr=self.params.lr)

        # optimizer.param_groups[0]['capturable'] = True
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.3, step_size=5)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, cycle_momentum=False, base_lr=1e-5,
                                                      max_lr=1e-3, step_size_up=10000, step_size_down=10000)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=16,eta_min=1e-4)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """training"""
        x, y, item, img_ori = batch
        x, y = x.to(device), y.to(device)

        output_feature, output_fc = self.model.forward(x)
        output = torch.squeeze(output_fc, dim=-1)

        loss_fn = self.loss_fn(output, y)
        predicts = F.softmax(output, dim=-1)
        predicts = torch.max(predicts, dim=-1)[1]

        acc = (predicts == y).sum().data * 1.0 / self.params.batch_size

        self.log('train_loss', loss_fn, batch_size=batch[0].shape[0], prog_bar=True)
        self.log('train_acc', acc, batch_size=batch[0].shape[0], prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], batch_size=batch[0].shape[0], prog_bar=True)

        # Sample Viz
        tumor_index = torch.where(y > 0)[0]
        for index in tumor_index:
            if np.random.randint(1, 50) == 3:
                corr = (predicts[index] == 1)
                self._viz_sample(img_ori, y, index, viz=viz_tumor_train, corr=corr)

        normal_index = torch.where(y == 0)[0]
        for index in normal_index:
            if np.random.randint(1, 50) == 3:
                corr = (predicts[index] == 0)
                self._viz_sample(img_ori, y, index, viz=viz_normal_train, corr=corr)
        return {'loss': loss_fn, 'train_acc': acc}
        
    def validation_step(self, batch, batch_idx):        
        x, y, item, img_ori = batch
        x, y = x.to(device), y.to(device)
        
        # 特征和标签
        output_feature, output_fc = self.model.forward(x)
        output = torch.squeeze(output_fc, dim=-1)

        loss_fn = self.loss_fn(output, y)

        predicts = F.softmax(output, dim=-1)
        predicts = torch.max(predicts, dim=-1)[1]
        pred_acc_bool = (predicts == y)
        
        # 总 acc
        acc = pred_acc_bool.type(torch.float).sum().data * 1.0 / self.params.batch_size

        # 分别计算每个类别的准确性
        all_labes = get_label_cate(mode=self.params.mode)
        results = []
        for label in all_labes:
            label_num = get_label_cate_num(label, mode=self.params.mode)
            pred_index = torch.where(predicts == label_num)[0]
            acc_cnt = torch.sum(y[pred_index] == label_num)
            fail_cnt = torch.sum(y[pred_index] != label_num)
            label_cnt = torch.sum(y == label_num)
            results.append([label, acc_cnt.cpu().item(), fail_cnt.cpu().item(), label_cnt.cpu().item()])

        # Sample Viz
        tumor_index = torch.where(y > 0)[0]
        for index in tumor_index:
            if np.random.randint(1, 50) == 3:
                corr = (predicts[index] == 1)
                self._viz_sample(img_ori, y, index, viz=viz_tumor_valid, corr=corr)

        normal_index = torch.where(y == 0)[0]
        for index in normal_index:
            if np.random.randint(1, 50) == 3:
                corr = (predicts[index] == 0)
                self._viz_sample(img_ori, y, index, viz=viz_normal_valid, corr=corr)

        results = np.array(results)

        if self.results is None:
            self.results = results
        else:
            self.results = np.concatenate((self.results, results), axis=0)

        self.log('val_loss', loss_fn, batch_size=batch[0].shape[0], prog_bar=True)
        self.log('val_acc', acc, batch_size=batch[0].shape[0], prog_bar=True)

        return {'val_loss': loss_fn, 'val_acc': acc}

    def _viz_sample(self, x, y, index, viz=None, corr=False):
        ran_idx = np.random.randint(1, 5)
        win = "win_{}".format(ran_idx)
        label = y[index]
        sample_img = x[index]
        title = "label:{}:,corr:{}".format(label, corr)
        visdom_data(sample_img, [], viz=viz, win=win, title=title)

    def on_validation_epoch_start(self):
        self.results = None

    def on_validation_epoch_end(self):
        columns = ["label", "acc_cnt", "fail_cnt", "real_cnt"]
        results_pd = pd.DataFrame(self.results, columns=columns)

        all_labes = get_label_cate(mode=self.params.mode)
        for label in all_labes:
            acc_cnt = results_pd[results_pd["label"] == label]["acc_cnt"].sum()
            fail_cnt = results_pd[results_pd["label"] == label]["fail_cnt"].sum()
            real_cnt = results_pd[results_pd["label"] == label]["real_cnt"].sum()
            # self.log('acc_cnt_{}'.format(label), float(acc_cnt), prog_bar=True)
            # self.log('fail_cnt_{}'.format(label), float(fail_cnt), prog_bar=True)
            # self.log('real_cnt_{}'.format(label), float(real_cnt), prog_bar=True)
            if acc_cnt + fail_cnt == 0:
                acc = 0.0
            else:
                acc = acc_cnt / (acc_cnt + fail_cnt)
            if real_cnt == 0:
                recall = 0.0
            else:
                recall = acc_cnt / real_cnt
            self.log('acc_{}'.format(label), acc, prog_bar=True)
            self.log('recall_{}'.format(label), recall, prog_bar=True)

    def train_dataloader(self):
        file_path = self.params.data_path
        tumor_mask_path = self.params.tumor_mask_path
        csv_path = os.path.join(file_path, self.params.train_csv)
        split_data = pd.read_csv(csv_path).values[:, 0].tolist()
        wsi_path = os.path.join(file_path, "data")
        mask_path = os.path.join(file_path, tumor_mask_path)

        # Data Aug
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=224, scale=(0.5, 0.8), ratio=(1, 5)),
            transforms.RandomHorizontalFlip(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # dataset_train = Whole_Slide_Bag_COMBINE(file_path, wsi_path, mask_path, work_type="train", mode=self.params.mode,
        #                                         patch_path=self.params.patch_path, transform=trans,
        #                                         patch_size=self.params.image_size, split_data=split_data,
        #                                         patch_level=self.params.patch_level)

        dataset_train = Whole_Slide_Bag_COMBINE_all(file_path, wsi_path, mask_path, work_type="train",
                                                    transform=trans,
                                                    image_size=self.params.image_size, 
                                                    split_data=split_data,
                                                    patch_level=self.params.patch_level)

        train_loader = DataLoader(dataset_train,
                                  batch_size=self.params.batch_size,
                                  collate_fn=self._collate_fn,
                                  shuffle=True,
                                  num_workers=self.params.num_workers,
                                  pin_memory=True)
        return train_loader

    def val_dataloader(self):
        file_path = self.params.data_path
        tumor_mask_path = self.params.tumor_mask_path
        csv_path = os.path.join(file_path, self.params.valid_csv)
        split_data = pd.read_csv(csv_path).values[:, 0].tolist()
        wsi_path = os.path.join(file_path, "data")
        mask_path = os.path.join(file_path, tumor_mask_path)

        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # dataset_valid = Whole_Slide_Bag_COMBINE(file_path, wsi_path, mask_path, work_type="valid", mode=self.params.mode,
        #                                         patch_path=self.params.patch_path, transform=trans,
        #                                         patch_size=self.params.image_size, split_data=split_data,
        #                                         patch_level=self.params.patch_level)

        dataset_valid = Whole_Slide_Bag_COMBINE_all(file_path, wsi_path, mask_path, work_type="val",
                                                    transform=trans,
                                                    image_size=self.params.image_size, 
                                                    split_data=split_data,
                                                    patch_level=self.params.patch_level)

        val_loader = DataLoader(dataset_valid,
                                batch_size=self.params.batch_size,
                                collate_fn=self._collate_fn,
                                shuffle=False,
                                num_workers=self.params.num_workers,
                                pin_memory=True)
        return val_loader

    def _collate_fn(self, batch):
        # 再次进行批处理数据
        aggregated = []
        for i in range(len(batch[0])):
            if i == 0:
                # 训练图片
                sample_list = [sample[i] for sample in batch]
                aggregated.append(torch.stack(sample_list, dim=0))
            elif i == 1:
                # 图片标签
                sample_list = [sample[i] for sample in batch]
                aggregated.append(torch.from_numpy(np.array(sample_list)))
            else:
                # 图像信息/原始图片
                aggregated.append([sample[i] for sample in batch])
        return aggregated


def get_last_ck_file(checkpoint_path):
    list = os.listdir(checkpoint_path)
    list.sort(key=lambda fn: os.path.getmtime(checkpoint_path + "/" + fn) if not os.path.isdir(
        checkpoint_path + "/" + fn) else 0)
    return list[-1]


def main(hparams):
    checkpoint_path = os.path.join(hparams.work_dir, "checkpoints", hparams.model_name)
    print("checkpoint_path: ", checkpoint_path)

    logger_name = "app_log"
    log_path = os.path.join(hparams.work_dir, logger_name, hparams.model_name)
    print("log_path: ", log_path)

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    filename = 'slfcd-{epoch:02d}-val_acc-{val_acc:.2f}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=checkpoint_path,
        filename=filename,
        save_last=True,
        save_top_k=-1,
        auto_insert_metric_name=False
    )

    model_logger = (
        pl_loggers.TensorBoardLogger(save_dir=hparams.work_dir, name=logger_name, version=hparams.model_name)
    )

    if hparams.load_weight:
        checkpoint_path_file = '/home/bavon/project/SLFCD/SLFCD/results/checkpoints/hsil_cbam_with_feature/slfcd-13-val_acc-0.82.ckpt'
        model = CoolSystem.load_from_checkpoint(checkpoint_path_file).to(device)
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            gpus=[int(hparams.device_ids)],
            accelerator='gpu',
            logger=model_logger,
            callbacks=[checkpoint_callback],
            log_every_n_steps=1
        )
        trainer.fit(model.to(device), ckpt_path=checkpoint_path_file)
    else:
        model = CoolSystem(hparams)
        model = model.to(device)
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            gpus=[int(hparams.device_ids)],
            accelerator='gpu',
            logger=model_logger,
            callbacks=[checkpoint_callback],
            log_every_n_steps=1
        )
        trainer.fit(model)


def data_summarize(dataloader):
    it = iter(dataloader)
    size = len(dataloader)

    viz_number_tumor, viz_number_normal = 0, 0
    label_stat = []
    for index in range(size):
        img, label, img_ori, item = next(it)
        img_ori = img_ori.cpu().numpy()[0]
        type = item['type'][0]
        if type == "annotation":
            label_stat.append(item['label'])
            if viz_number_tumor < 10:
                viz_number_tumor += 1
                visdom_data(img_ori, [], title="tumor_{}".format(index), viz=viz_tumor_valid)
        else:
            label_stat.append(0)
            if viz_number_normal < 10:
                viz_number_normal += 1
                visdom_data(img_ori, [], title="normal_{}".format(index), viz=viz_normal_valid)

    label_stat = np.array(label_stat)
    print("label_stat 1:{},2:{},3:{}".format(np.sum(label_stat == 1), np.sum(label_stat == 2), np.sum(label_stat == 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--device_ids', default='1', type=str, help='choose device')
    parser.add_argument('--mode', default='ais', type=str, help='choose type')

    args = parser.parse_args()
    device_ids = args.device_ids

    if args.mode == "hsil":
        cnn_path = 'configs/config_hsil_lc.json'
    if args.mode == "lsil":
        cnn_path = 'configs/config_lsil_lc.json'
    if args.mode == "ais":
        cnn_path = 'configs/config_ais_lc.json'
    with open(cnn_path, 'r') as f:
        args = json.load(f)

    args["device_ids"] = device_ids
    hyperparams = Namespace(**args)
    print("hyperparams: ", hyperparams)
    main(hyperparams)
    print('process success!!!')
    
