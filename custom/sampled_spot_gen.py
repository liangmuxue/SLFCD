import os
import sys
import logging
import argparse

import numpy as np

sys.path.append(os.path.join(os.path.abspath(__file__), "/../../"))


parser = argparse.ArgumentParser(description="Get center points of patches "
                                             "from mask")
parser.add_argument("mask_path", default=None, metavar="MASK_PATH", type=str,
                    help="Path to the mask npy file")
parser.add_argument("txt_path", default=None, metavar="TXT_PATH", type=str,
                    help="Path to the txt file")
parser.add_argument("patch_number", default=None, metavar="PATCH_NUMB", type=int,
                    help="The number of patches extracted from WSI")
parser.add_argument("--level", default=6, metavar="LEVEL", type=int,
                    help="Bool format, whether or not")


class patch_point_in_mask_gen(object):
    '''
    extract centre point from mask
    inputs: mask path, centre point number
    outputs: centre point
    '''

    def __init__(self, mask_path, number):
        self.mask_path = mask_path
        self.number = number

    def get_patch_point(self):
        mask_tissue = np.load(self.mask_path)
        X_idcs, Y_idcs = np.where(mask_tissue)

        centre_points = np.stack(np.vstack((X_idcs.T, Y_idcs.T)), axis=1)

        if centre_points.shape[0] > self.number:
            sampled_points = centre_points[np.random.randint(centre_points.shape[0],
                                                             size=self.number), :]
        else:
            sampled_points = centre_points
        return sampled_points


def run(mask_path,txt_path,mode="tumor",patch_number=1000,level=2,ignore_build=False):
    
    mask_file_names = os.listdir(mask_path)
    train_sp_size = len(mask_file_names)*0.7
    index = 0
    type = "train"
    if not ignore_build:
        for mask_file in mask_file_names:
            if index>train_sp_size:
                type = "valid"
            mask_name = os.path.split(mask_file)[-1].split(".")[0]
            txt_file_path = "{}/{}/{}_{}.txt".format(txt_path,type,mask_name,mode)
            mask_file_path = os.path.join(mask_path,mask_file)  
            sampled_points = patch_point_in_mask_gen(mask_file_path, patch_number).get_patch_point()
            sampled_points = (sampled_points * 2 ** level).astype(np.int32) # make sure the factor
            name = np.full((sampled_points.shape[0], 1), mask_name)
            center_points = np.hstack((name, sampled_points))
            
            with open(txt_file_path, "a") as f:
                np.savetxt(f, center_points, fmt="%s", delimiter=",")
            index+=1
            
        
def MergeTxt(filepath,outfile,contain_filter=None):
    k = open(outfile, 'a+')
    for parent, dirnames, filenames in os.walk(filepath):
        for filepath in filenames:
            if contain_filter is not None and contain_filter not in filepath:
                continue
            txtPath = os.path.join(parent, filepath) 
            f = open(txtPath)
            k.write(f.read()+"\n")

    k.close()

def main():
    logging.basicConfig(level=logging.INFO)
    for mode in ["tumor","normal"]:
        for type in ["lsil","hsil"]:
            mask_path = "/home/bavon/datasets/wsi/{}/{}_mask".format(type,mode)
            txt_path = "/home/bavon/datasets/wsi/{}/txt".format(type)
            run(mask_path,txt_path,mode=mode,ignore_build=False)


def merge():
    for type in ["lsil","hsil"]:
        for task_type in ["train","valid"]:
            for tum_type in ["normal","tumor"]:
                txt_path = "/home/bavon/datasets/wsi/{}/txt/{}".format(type,task_type)
                outfile = "/home/bavon/datasets/wsi/{}/txt/{}_{}_total.txt".format(type,task_type,tum_type)                
                MergeTxt(txt_path,outfile,contain_filter="_{}".format(tum_type))
        
if __name__ == "__main__":
    main()
    merge()
