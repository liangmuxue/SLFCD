import sys
import os
import argparse
import logging

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

parser = argparse.ArgumentParser(description="Get the normal region"
                                             " from tumor WSI ")
parser.add_argument("tumor_path", default=None, metavar='TUMOR_PATH', type=str,
                    help="Path to the tumor mask npy")
parser.add_argument("tissue_path", default=None, metavar='TISSUE_PATH', type=str,
                    help="Path to the tissue mask npy")
parser.add_argument("normal_path", default=None, metavar='NORMAL_PATCH', type=str,
                    help="Path to the output normal region from tumor WSI npy")


def run(tumor_path,tissue_path,normal_path,level=2):
    
    for tumor_file in os.listdir(tumor_path):
        tumor_file_path = tumor_path + "/" + tumor_file
        tumor_mask = np.load(tumor_file_path)
        normal_file_path = normal_path + "/" + tumor_file
        tissue_file_path = tissue_path + "/" + tumor_file
        tissue_mask = np.load(tissue_file_path)
    
        normal_mask = tissue_mask & (~ tumor_mask)
    
        np.save(normal_file_path, normal_mask)

def main():
    logging.basicConfig(level=logging.INFO)
    type_part = "lsil"
    # type_part = "hsil"
    tissue_path = "/home/bavon/datasets/wsi/{}/mask".format(type_part)
    tumor_path = "/home/bavon/datasets/wsi/{}/tumor_mask".format(type_part)   
    normal_path = "/home/bavon/datasets/wsi/{}/normal_mask".format(type_part)        
    run(tumor_path,tissue_path,normal_path)


if __name__ == "__main__":
    main()
