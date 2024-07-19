import sys
sys.path.append(r'/home/bavon/project/SLFCD/SLFCD/')
sys.path.append(r'/home/bavon/project/SLFCD/SLFCD/extras/')
sys.path.append(r'/home/bavon/project/SLFCD/SLFCD/project/')
import os
import argparse

import json
from argparse import Namespace
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import pickle

from clam.datasets.dataset_inference import Whole_Slide_Bag_Infer
from utils.wsi_img_viz import viz_infer_dataset
from custom.train_with_clamdata import CoolSystem

device = torch.device('cuda:1')


def load_model(hparams, device_ids):
    checkpoint_path = os.path.join("..", "..", hparams.work_dir, "checkpoints", hparams.model_name)
    filename = 'slfcd-{epoch:02d}-val_acc-{val_acc:.2f}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=checkpoint_path,
        filename=filename,
        save_top_k=-1,
        auto_insert_metric_name=False
    )

    logger_name = "app_log"
    model_logger = (
        pl_loggers.TensorBoardLogger(save_dir=hparams.work_dir, name=logger_name, version=hparams.model_name)
    )

    checkpoint_path_file = '/home/bavon/project/SLFCD/SLFCD/results/checkpoints/ais_cbam_with_feature718/slfcd-06-val_acc-0.89.ckpt'
    print("checkpoint_path: ", checkpoint_path_file)

    model = CoolSystemInfer.load_from_checkpoint(checkpoint_path_file).to(device)
    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        gpus=[int(device_ids)],
        accelerator='gpu',
        logger=model_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1
    )

    model.params = hparams

    return model, trainer


def main(model, trainer, dataset_infer, single_name=None, result_path=None):
    model.result_path = result_path
    model.file_name = single_name

    inference_loader = DataLoader(dataset_infer, batch_size=64, collate_fn=model._collate_fn,
                                  shuffle=False, num_workers=1)
    print("total len:{}".format(len(dataset_infer)))
    trainer.predict(model=model, dataloaders=inference_loader)


def viz_results(dataset_infer, single_name=None, result_path=None):
    """可视化"""
    save_path = os.path.join(result_path, single_name, "ret_{}.pkl".format(single_name))
    loader = open(save_path, 'rb')
    result_data = pickle.load(loader)
    viz_infer_dataset(result_data, dataset=dataset_infer, result_path=result_path)


def single_img_inference(img_path, model, hparams=None):
    # 使用当前配置里的超参数
    model.params = hparams
    model.result_path = result_path
    img = cv2.imread(img_path)
    model.single_inference(img)


class CoolSystemInfer(CoolSystem):
    def __init__(self, hparams, file_name=None):
        super(CoolSystemInfer, self).__init__(hparams)
        self.file_name = file_name
        self.results = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """根据给出的数据文件，判断patch区域所属类别"""
        img_tar, coord, item, img = batch
        batch_size = img_tar.shape[0]
        results = np.zeros((batch_size, 2))

        img_tar = img_tar.to(device)
        feature, cls_out = self.model(img_tar)
        probs = F.softmax(cls_out, dim=-1)
        predicts = torch.max(probs, dim=-1)[1]
        tum_idx = torch.where(predicts == 1)[0]
        if tum_idx.shape[0] > 0:
            probs = probs.cpu().numpy()
            tum_idx = tum_idx.cpu().numpy()
            # print("has pred 1:{}".format(tum_idx))
            img_ori = np.array(img)[tum_idx]
            for j in range(img_ori.shape[0]):
                probs_show = int(round(probs[tum_idx[j]][1], 2) * 100)
                item[tum_idx[j]]["pred"] = 1
                item[tum_idx[j]]["probs"] = probs_show
            results[tum_idx, 0] = 1
            results[tum_idx, 1] = probs[tum_idx, 1]

        self.results.append(item)

    def on_predict_end(self):
        results = np.concatenate(np.array(self.results))
        save_path = os.path.join(self.result_path, self.file_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        writer = open(os.path.join(save_path, "ret_{}.pkl".format(self.file_name)), 'wb')
        pickle.dump(results, writer)
        writer.close()

    def single_inference(self, img):
        img = cv2.resize(img, (224, 224))
        img_tar = torch.Tensor(img).to(device).permute(2, 0, 1).unsqueeze(0)
        feature, cls_out = self.model(img_tar)
        probs = F.softmax(cls_out, dim=-1)
        predicts = torch.max(probs, dim=-1)[1]
        print("predicts:{},probs:{}".format(predicts, probs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--device_ids', default='1', type=str, help='choose device')
    parser.add_argument('--mode', default='ais', type=str, help='choose type')
    parser.add_argument('--result_path', default='/home/bavon/project/SLFCD/SLFCD/custom/infer/ais_cbam_with_feature_infer', type=str)
    parser.add_argument('--slide_size', default=256, type=int)
    parser.add_argument('--inf_filename', default='/home/bavon/datasets/wsi/ais/data/76-CG21_00825_02.svs', type=str)
    args = parser.parse_args()

    device_ids = args.device_ids
    result_path = args.result_path
    slide_size = args.slide_size
    inf_filename = args.inf_filename
    single_name = os.path.split(inf_filename)[-1][:-4]
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        
    if args.mode == "hsil":
        cnn_path = '../configs/config_hsil_lc.json'
    if args.mode == "lsil":
        cnn_path = '../configs/config_lsil_liang.json'
    if args.mode == "ais":
        cnn_path = '../configs/config_ais_lc.json'
    with open(cnn_path, 'r') as f:
        args = json.load(f)
    args['data_path'] = os.path.split(inf_filename)[0][:-5]
    
    hyperparams = Namespace(**args)
    print("hyperparams: ", hyperparams)
    model, trainer = load_model(hyperparams, device_ids)
    
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
    ])
    
    # dataset_infer = Whole_Slide_Bag_Infer(hyperparams.data_path, single_name, patch_size=hyperparams.image_size,
    #                           slide_size=slide_size, patch_level=hyperparams.patch_level, transform=trans)
    #
    # main(model, trainer, dataset_infer, single_name=single_name, result_path=result_path)
    # viz_results(dataset_infer, single_name=single_name, result_path=result_path)
    
    for single_name in os.listdir('/home/bavon/datasets/wsi/ais/data'):
        single_name = single_name[:-4]
        if single_name in os.listdir('/home/bavon/project/SLFCD/SLFCD/custom/infer/ais_cbam_with_feature_infer/'):
            continue
        print(f"--------------- {single_name} ----------------------")
        dataset_infer = Whole_Slide_Bag_Infer(hyperparams.data_path, single_name, patch_size=hyperparams.image_size,
                                      slide_size=slide_size, patch_level=hyperparams.patch_level, transform=trans)
    
        main(model, trainer, dataset_infer, single_name=single_name, result_path=result_path)
        viz_results(dataset_infer, single_name=single_name, result_path=result_path)
    
    # img_path = "/home/bavon/project/SLFCD/SLFCD/custom/infer/ais_cbam_with_feature_infer/76-CG21_00825_02/FirstPhase/2688_2944_2303_2559.jpg"
    # single_img_inference(img_path, model, hparams=hyperparams)
    print("process successful!!!")
