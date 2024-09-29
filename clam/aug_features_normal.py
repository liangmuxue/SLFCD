import sys
sys.path.append("/home/bavon/project/SLFCD/SLFCD/extras/")
sys.path.append("/home/bavon/project/SLFCD/SLFCD/project/")
sys.path.append("/home/bavon/project/SLFCD/SLFCD/")
import torch
import os
import argparse
import random
import pandas as pd
import csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--types', type=str, default='normal')
    parser.add_argument('--feat_dir', type=str, default='/home/bavon/datasets/wsi/combine3')
    args = parser.parse_args()

    csv_path = os.path.join(args.feat_dir, "train.csv")
    slide_data = pd.read_csv(csv_path, usecols=['slide_id', 'label'])
    label_0_df = slide_data[slide_data['label'] == 0]
    
    shuffled_features_file = []
    # for i in range(2):
    for i in range(2, len(label_0_df)):
        svs_1 = label_0_df.iloc[i - 2]['slide_id'][:-4]
        svs_2 = label_0_df.iloc[i - 1]['slide_id'][:-4]
        svs_3 = label_0_df.iloc[i]['slide_id'][:-4]
        print('process ', svs_1, svs_2, svs_3)
        
        full_path = os.path.join(args.feat_dir, 'features/pt_files', args.types, '{}.pt'.format(svs_1))
        svs_1_features = torch.load(full_path)
            
        full_path = os.path.join(args.feat_dir, 'features/pt_files', args.types, '{}.pt'.format(svs_2))
        svs_2_features = torch.load(full_path)
        
        full_path = os.path.join(args.feat_dir, 'features/pt_files', args.types, '{}.pt'.format(svs_3))
        svs_3_features = torch.load(full_path)
        
        all_features = torch.cat((svs_1_features, svs_2_features, svs_3_features), dim=0)
        
        for i, num in zip(range(5), [random.randint(min(svs_3_features.size(0), svs_2_features.size(0)), max(svs_3_features.size(0), svs_2_features.size(0))), 
                                     random.randint(min(svs_2_features.size(0), svs_1_features.size(0)), max(svs_2_features.size(0), svs_1_features.size(0))),
                                     svs_3_features.size(0), svs_2_features.size(0), svs_1_features.size(0)]):
            indices = torch.randperm(all_features.size(0))[:num]
            shuffled_features = all_features[indices]
            
            torch.save(shuffled_features, os.path.join(args.feat_dir, 'features/pt_files', args.types, 
                                                       f'{svs_1.split("_")[0]}_{svs_2.split("_")[0]}_{svs_3.split("_")[0]}_{i}.pt'))
            
            shuffled_features_file.append([100, 100, f'{svs_1.split("_")[0]}_{svs_2.split("_")[0]}_{svs_3.split("_")[0]}_{i}.svs', 0, 'normal'])
            
        # shuffled_df = label_0_df.sample(frac=1).reset_index(drop=True)
        # label_0_df = shuffled_df
        
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(shuffled_features_file)
    
    print("process successful!!!", len(shuffled_features_file))
