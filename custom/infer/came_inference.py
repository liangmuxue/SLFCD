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

from clam.datasets.dataset_inference import Whole_Slide_Bag_Infer, Whole_Slide_Bag_Infer_all
from utils.wsi_img_viz import viz_infer_dataset
from custom.train_with_clamdata import CoolSystem
from tqdm import tqdm
from utils.vis import put_mask
import openslide
import h5py
import csv

device = torch.device('cuda:1')


def load_model(hparams, checkpoint_path_file, device_ids):
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
    model.eval()
    return model, trainer


def viz_results(dataset_infer, conf, result_path=None):
    """可视化"""
    single_name = dataset_infer.single_name
    image = dataset_infer.image
    total_img_copy1 = image.copy()
    total_img_copy2 = image.copy()
    save_path = os.path.join(result_path, single_name, "ret_{}.pkl".format(single_name))
    loader = open(save_path, 'rb')
    results = pickle.load(loader)
    # viz_infer_dataset(result_data, total_img=dataset_infer.image, single_name=single_name,
    #                   patch_size=dataset_infer.patch_size, result_path=result_path)
    viz_infer_show(probs=results["probs"], coords=results["coords"], conf=conf,
                   image=image, total_img_copy1=total_img_copy1, total_img_copy2=total_img_copy2,
                   single_name=single_name, patch_size=patch_size, result_path=result_path)


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
        self.results = {'probs': [], 'coords': []}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """根据给出的数据文件，判断patch区域所属类别"""
        img_tar, coord_tar = batch
        img_tar = img_tar.to(device)
        feature, cls_out = self.model(img_tar)

        output = torch.squeeze(cls_out, dim=-1)
        probs = F.softmax(output, dim=-1)
        predicts = torch.max(probs, dim=-1)

        probs = predicts.values.cpu()
        label = predicts.indices.cpu()

        index = torch.where(label == 1)[0]
        probs_1 = probs[index].tolist()
        # label_1 = label[index]
        coord_tar_1 = coord_tar[index].tolist()

        self.results["probs"].extend(probs_1)
        self.results["coords"].extend(coord_tar_1)

    def on_predict_end(self):
        save_path = os.path.join(self.result_path, self.file_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        writer = open(os.path.join(save_path, "ret_{}.pkl".format(self.file_name)), 'wb')
        pickle.dump(self.results, writer)
        writer.close()


def single_inference(img, model, trans):
    img = cv2.resize(img, (224, 224))
    img = trans(img)
    # img = torch.stack([img, img], dim=1)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        output_feature, output_fc = model(img)
    output = torch.squeeze(output_fc, dim=-1)
    probs = F.softmax(output, dim=-1)
    predicts = torch.max(probs, dim=-1)

    probs = predicts.values.cpu().tolist()
    label = predicts.indices.cpu().tolist()[0]
    # print("predicts:{},probs:{:.2f}".format(label, probs[0]))
    return probs, label


def viz_infer_show(probs, coords, conf, image, total_img_copy1, total_img_copy2, single_name=None, patch_size=None,
                   result_path=None):
    # 标注颜色
    color_value = (85, 85, 205)

    index = np.where(np.array(probs) > conf)[0]
    probs = np.array(probs)[index].tolist()
    coords = np.array(coords)[index].tolist()

    for coord, prob in tqdm(zip(coords, probs), total=len(coords), desc=f"save img {single_name}"):
        region = [coord[0], coord[1], coord[0] + patch_size, coord[1] + patch_size]
        cv2.rectangle(total_img_copy2, (region[0], region[1]), (region[2], region[3]), (255, 0, 0), 10, 5)

        try:
            img_ori = total_img_copy1[region[1]:region[3], region[0]:region[2], :]
            img = cv2.resize(img_ori, (patch_size, patch_size))
            if not os.path.exists(f"{result_path}/{single_name}/FirstPhase"):
                os.makedirs(f"{result_path}/{single_name}/FirstPhase")
            cv2.imwrite(
                f"{result_path}/{single_name}/FirstPhase/{region[0]}_{region[1]}_{region[2]}_{region[3]}_{prob:.2f}.jpg",
                img)

            x_min = region[0]
            x_max = region[2]
            y_min = region[1]
            y_max = region[3]
            if y_max > image.shape[0] or y_min > image.shape[0]:
                continue
            if x_max > image.shape[1] or x_min > image.shape[1]:
                continue
            image = put_mask(image, region, color=color_value)
            image[y_min:y_max, x_min, :] = color_value
            image[y_min:y_max, x_max, :] = color_value
            image[y_min, x_min:x_max, :] = color_value
            image[y_max, x_min:x_max, :] = color_value
        except:
            pass
    save_path_1 = os.path.join(result_path, single_name, "plt_{}.jpg".format(single_name))
    save_path_2 = os.path.join(result_path, single_name, "plt_copy_{}.jpg".format(single_name))
    cv2.imwrite(save_path_1, image)
    cv2.imwrite(save_path_2, total_img_copy2)

def batch_img(hyperparams, inf_csv, checkpoint_path_file, device_ids, trans, patch_size, slide_size, result_path, single_name):
    model, trainer = load_model(hyperparams, checkpoint_path_file, device_ids)
    # dataset_infer = Whole_Slide_Bag_Infer(hyperparams.data_path, single_name, patch_size=hyperparams.image_size,
    #                                       slide_size=slide_size, patch_level=hyperparams.patch_level,
    #                                       transform=trans,
    #                                       result_path=result_path)

    csv_reader = csv.reader(open(inf_csv))
    next(csv_reader)
    for row in csv_reader:
        single_name = row[0][:-4]
        print(f"------------------------ {single_name} -------------------------------")
        if not os.path.exists(result_path + "/" + single_name):
            os.makedirs(result_path + "/" + single_name)
            
        dataset_infer = Whole_Slide_Bag_Infer_all(hyperparams.data_path, single_name,
                                                  patch_size=patch_size,
                                                  slide_size=slide_size,
                                                  patch_level=hyperparams.patch_level,
                                                  transform=trans)
        inference_loader = DataLoader(dataset_infer, batch_size=hyperparams.batch_size, shuffle=False,
                                      num_workers=hyperparams.num_workers)
        model.result_path = result_path
        model.file_name = single_name
        trainer.predict(model=model, dataloaders=inference_loader)
        viz_results(dataset_infer, conf=conf, result_path=result_path)


def single_img(hyperparams, checkpoint_path_file, trans, inf_csv, conf):
    model = CoolSystemInfer.load_from_checkpoint(checkpoint_path_file).to(device)
    model.eval()
    csv_reader = csv.reader(open(inf_csv))
    next(csv_reader)
    for row in csv_reader:
        single_name = row[0][:-4]
        print(f"------------------------ {single_name} -------------------------------")
        if not os.path.exists(result_path + "/" + single_name):
            os.makedirs(result_path + "/" + single_name)

        dataset_infer = Whole_Slide_Bag_Infer_all(hyperparams.data_path, single_name, patch_size=hyperparams.image_size,
                                          slide_size=slide_size, patch_level=hyperparams.patch_level,
                                          transform=trans,
                                          result_path=result_path)
        
        inference_loader = DataLoader(dataset_infer, batch_size=hyperparams.batch_size, shuffle=False,
                              num_workers=hyperparams.num_workers)
                
        image = dataset_infer.image
        total_img_copy1 = image.copy()
        total_img_copy2 = image.copy()
        results = {'probs': [], 'coords': []}
        
        with torch.no_grad():
            for img_tar, coord_tar in tqdm(inference_loader, desc="infer svs"):
                img_tar = img_tar.to(device)
                output_feature, output_fc = model(img_tar)
            
                output = torch.squeeze(output_fc, dim=-1)
                probs = F.softmax(output, dim=-1)
                predicts = torch.max(probs, dim=-1)
        
                probs = predicts.values.cpu()
                label = predicts.indices.cpu()
        
                index = torch.where(label == 1)[0]
                probs_1 = probs[index].tolist()
                # label_1 = label[index]
                coord_tar_1 = coord_tar[index].tolist()
        
                results["probs"].extend(probs_1)
                results["coords"].extend(coord_tar_1)
        
        save_path = os.path.join(result_path, single_name, "ret_{}.pkl".format(single_name))
        writer = open(save_path, 'wb')
        pickle.dump(results, writer)
        writer.close()

        # loader = open(save_path, 'rb')
        # results = pickle.load(loader)

        viz_infer_show(probs=results["probs"], coords=results["coords"], conf=conf,
                       image=image, total_img_copy1=total_img_copy1, total_img_copy2=total_img_copy2,
                       single_name=single_name, patch_size=patch_size, result_path=result_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='infer model')
    parser.add_argument('--device_ids', default='1', type=str, help='choose device')
    parser.add_argument('--mode', default='hsil', type=str, help='choose type')
    parser.add_argument('--result_path', default=r'/home/bavon/project/SLFCD/SLFCD/custom/infer/hsil_cbam_with_feature_infer1',
                        type=str)
    parser.add_argument('--patch_size', default=256, type=int)
    parser.add_argument('--slide_size', default=128, type=int)
    parser.add_argument('--conf', default=0, type=float)
    parser.add_argument('--inf_weights',
                        default=r"/home/bavon/project/SLFCD/SLFCD/results/checkpoints/hsil_cbam_with_feature/slfcd-09-val_acc-0.93.ckpt",
                        type=str)
    parser.add_argument('--inf_filename', default=r"/home/bavon/datasets/wsi/hsil/data/99-CG20_01009_05.svs", type=str)
    parser.add_argument('--inf_csv', default=r"/home/bavon/datasets/wsi/hsil/valid.csv", type=str)
    args = parser.parse_args()

    device_ids = args.device_ids
    result_path = args.result_path
    patch_size = args.patch_size
    slide_size = args.slide_size
    inf_filename = args.inf_filename
    inf_csv = args.inf_csv
    single_name = os.path.split(inf_filename)[-1][:-4]
    checkpoint_path_file = args.inf_weights
    conf = args.conf
    print("checkpoint_path: ", checkpoint_path_file)

    if args.mode == "hsil":
        cnn_path = '../configs/config_hsil_lc.json'
    if args.mode == "lsil":
        cnn_path = '../configs/config_lsil_liang.json'
    if args.mode == "ais":
        cnn_path = '../configs/config_ais_lc.json'
    with open(cnn_path, 'r') as f:
        args = json.load(f)

    hyperparams = Namespace(**args)
    print("hyperparams: ", hyperparams)

    save_path = os.path.join(result_path, single_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(224)])

    # single_img(hyperparams, checkpoint_path_file, trans, inf_csv, conf)
    
    batch_img(hyperparams, inf_csv, checkpoint_path_file, device_ids, trans, patch_size, slide_size, result_path, single_name)
    

    print("process successful!!!")
