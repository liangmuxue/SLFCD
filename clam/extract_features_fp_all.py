import sys
sys.path.append("/home/bavon/project/SLFCD/SLFCD/extras/")
sys.path.append("/home/bavon/project/SLFCD/SLFCD/project/")
sys.path.append("/home/bavon/project/SLFCD/SLFCD/")
import shutil
import torch
import os
import torchvision.transforms as transforms
from clam.datasets.dataset_h5 import Dataset_Combine_Bags, Whole_Slide_Bag_FP_all
from torch.utils.data import DataLoader
import argparse
from utils.file_utils import save_hdf5
import h5py
import openslide
from torchvision import models
from torch import nn

device = torch.device('cuda:1')


def compute_w_loader(file_path, output_path, wsi, model, batch_size=8,print_every=20, patch_level=1, patch_size=256, slide_size=128):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
    ])

    dataset = Whole_Slide_Bag_FP_all(file_path, wsi, patch_level, patch_size, slide_size, transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, pin_memory=True, num_workers=4)

    mode = 'w'
    with torch.no_grad():
        for count, (batch, coords) in enumerate(loader):
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)

            features = model(batch)
            features = features.cpu().numpy()

            asset_dict = {'features': features, 'coords': coords.cpu().numpy()}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'

    # 保存为pt
    file = h5py.File(output_path, "r")
    features = file['features'][:]
    print('features size: ', features.shape)
    print('coordinates size: ', file['coords'].shape)
    features = torch.from_numpy(features)
    torch.save(features, fea_file_path)


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_dir', type=str, default='/home/bavon/datasets/wsi')
parser.add_argument('--patch_level', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--slide_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--types', type=str, default='ais,hsil,lsil,normal')
parser.add_argument('--feat_dir', type=str, default='/home/bavon/datasets/wsi/combine/features')
args = parser.parse_args()

if __name__ == '__main__':
    data_path = args.data_dir
    types = args.types.split(",")
    # 得到需要进行第二阶段训练的数据
    bags_dataset = Dataset_Combine_Bags(data_path, types)

    os.makedirs(args.feat_dir, exist_ok=True)
    for type in types:
        if not os.path.exists(os.path.join(args.feat_dir, 'pt_files', type)):
            # shutil.rmtree(os.path.join(args.feat_dir, 'pt_files', type))
            os.makedirs(os.path.join(args.feat_dir, 'pt_files', type), exist_ok=True)
        print("pt_files: ", os.path.join(args.feat_dir, 'pt_files', type))

        if not os.path.exists(os.path.join(args.feat_dir, 'h5_files', type)):
            # shutil.rmtree(os.path.join(args.feat_dir, 'h5_files', type))
            os.makedirs(os.path.join(args.feat_dir, 'h5_files', type), exist_ok=True)
        print("h5_files: ", os.path.join(args.feat_dir, 'h5_files', type))

    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
    print("dest_files: ", dest_files)

    import torch
    import numpy as np
    import random
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    model = models.resnet18(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2048)
    
    # 256 2048
    model = model.to(device)
    model_state_dict = torch.load('/home/bavon/project/SLFCD/SLFCD/clam/resnet18-f37072fd.pth')
    state_dict = {k: v for k, v in model_state_dict.items() if 'fc' not in k}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    for bag_candidate_idx in range(len(bags_dataset)):
        print('progress: {}/{}'.format(bag_candidate_idx, len(bags_dataset)))
        type, slide_id = bags_dataset[bag_candidate_idx]
        data_slide_dir = os.path.join(data_path, type, "data")
        data_h5_dir = os.path.join(data_path, type, "patches_level{}".format(args.patch_level))
        slide_id = slide_id.split(args.slide_ext)[0]

        bag_name = slide_id + '.h5'
        bag_base, _ = os.path.splitext(bag_name)
        fea_file_path = os.path.join(args.feat_dir, 'pt_files', type, bag_base + '.pt')

        h5_file_path = os.path.join(data_h5_dir, bag_name)
        slide_file_path = os.path.join(data_slide_dir, slide_id + args.slide_ext)

        output_path = os.path.join(args.feat_dir, 'h5_files', type, bag_name)
        print("output_path: ", output_path)

        wsi = openslide.open_slide(slide_file_path)
        try:
            compute_w_loader(h5_file_path, output_path, wsi, model=model,
                             batch_size=args.batch_size, print_every=20,
                             patch_level=args.patch_level,
                             patch_size=args.patch_size,
                             slide_size=args.slide_size)

        except Exception as e:
            print("compute_w_loader fail:{} {}".format(bag_name, e))
            continue
    print("process successful!!!")
