from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from clam.utils.file_utils import save_pkl, load_pkl
from clam.utils.core_utils import train
from clam.datasets.dataset_generic import Combine_MIL_Dataset

import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def return_splits(data_dir, type='ais'):
    train_dataset = Combine_MIL_Dataset(
        data_dir=data_dir,
        mode="train",
        shuffle=False,
        seed=args.seed,
        print_info=True,
        patient_strat=False,
        type=type,
        ignore=[])
    valid_dataset = Combine_MIL_Dataset(
        data_dir=data_dir,
        mode="valid",
        shuffle=False,
        seed=args.seed,
        print_info=True,
        patient_strat=False,
        type=type,
        ignore=[])
    test_dataset = Combine_MIL_Dataset(
        data_dir=data_dir,
        mode="test",
        shuffle=False,
        seed=args.seed,
        print_info=True,
        patient_strat=False,
        type=type,
        ignore=[])
    return train_dataset, valid_dataset, test_dataset


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = return_splits(args.data_dir, type=args.type)

        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        # write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        print("save_model_weights: ", filename)
        save_pkl(filename, results)
        break

    final_df = pd.DataFrame({'folds': [folds[0]], 'test_auc': all_test_auc,
                             'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc': all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    print("save_summary_path: ", os.path.join(args.results_dir, save_name))


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--type', default="ais", type=str)
parser.add_argument('--load_weights', action='store_true', default=False, help='Load pretrained weights to model')
parser.add_argument('--data_dir', type=str, default='/home/bavon/datasets/wsi/combine', help='data directory')
parser.add_argument('--max_epochs', type=int, default=100,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0) (训练标签的分数（默认值：1.0）)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='results', help='results directory (default: ./results)')
parser.add_argument('--log_data', action='store_true', default=True, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=True, help='enable dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                    help='slide-level classification loss function (default: ce) (滑动级别分类损失函数（默认：ce）)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb',
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results', default="combine")
parser.add_argument('--weighted_sample', action='store_true', default=False,
                    help='enable weighted sampling (使用加权抽样)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small',
                    help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'], default='task_1_tumor_vs_normal')
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                    help='disable instance-level clustering (禁用实例级群集)')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                    help='instance-level clustering loss function (default: None) (实例级聚类损失函数（默认：无）)')
parser.add_argument('--subtyping', action='store_true', default=False,
                    help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
args = parser.parse_args()

device = args.device


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
    settings.update({'bag_weight': args.bag_weight,
                     'inst_loss': args.inst_loss,
                     'B': args.B})

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    args.n_classes = 2
    print('\nLoad Dataset')

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    # args.results_dir = os.path.join("..", args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    args.results_dir = os.path.join("..", args.results_dir, str(args.exp_code), args.task + '_CLAM_50_' + args.type +'_s{}'.format(args.seed))
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
        print(settings, file=f)
    f.close()

    results = main(args)
    print("finished!")
    print("end script")
    print("process successful!!!") 
