import sys
import os
import argparse
import logging

import numpy as np
import openslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save'
                                 ' it in npy format')
parser.add_argument('--RGB_min', default=50, type=int, help='min value for RGB'
                    ' channel, default 50')


def run(wsi_path,npy_path,RGB_min=50,level=1):
    logging.basicConfig(level=logging.INFO)

    # walk all svs files,and convert
    for wsi_file in os.listdir(wsi_path):
        print("process file:",wsi_file)
        # if wsi_file!="23-CG23_12350_02.svs":
        #     continue
        if not wsi_file.endswith(".svs"):
            continue
        npy_file = wsi_file.replace(".svs", ".npy") 
        npy_file = npy_path + "/" + npy_file    
        slide = openslide.OpenSlide(wsi_path + "/" + wsi_file)
        if len(slide.level_dimensions)<=level:
            print("no level for {},ignore:".format(wsi_file))
            continue
        # note the shape of img_RGB is the transpose of slide.level_dimensions
        img_RGB = np.transpose(np.array(slide.read_region((0, 0),
                               level,
                               slide.level_dimensions[level]).convert('RGB')),
                               axes=[1, 0, 2])
    
        img_HSV = rgb2hsv(img_RGB)
    
        background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
        background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
        background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
        min_R = img_RGB[:, :, 0] > RGB_min
        min_G = img_RGB[:, :, 1] > RGB_min
        min_B = img_RGB[:, :, 2] > RGB_min
    
        tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
        np.save(npy_file, tissue_mask)
        


def main():
    wsi_path = "/home/bavon/datasets/wsi/test"
    npy_path = "/home/bavon/datasets/wsi/test/mask"    
    run(wsi_path,npy_path)


if __name__ == '__main__':
    main()
