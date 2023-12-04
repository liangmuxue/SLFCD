import os
import sys
import logging
import argparse

import numpy as np
import openslide
import cv2
import json

from utils.constance import get_label_with_group_code

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tumor mask of tumor-WSI and '
                                             'save it in npy format')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')
parser.add_argument('json_path', default=None, metavar='JSON_PATH', type=str,
                    help='Path to the JSON file')
parser.add_argument('npy_path', default=None, metavar='NPY_PATH', type=str,
                    help='Path to the output npy mask file')
parser.add_argument('--level', default=6, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')


def run(wsi_path,npy_path,json_path,level=0):
    
    for json_file in os.listdir(json_path):
        json_file_path = os.path.join(json_path,json_file)
        single_name = json_file.split(".")[0]
        npy_file = os.path.join(npy_path,single_name+".npy")
        wsi_file_path = os.path.join(wsi_path,single_name+".svs")
        slide = openslide.OpenSlide(wsi_file_path)
        if len(slide.level_dimensions)<=level:
            print("no level for {},ignore:".format(wsi_file_path))
            continue        
        w, h = slide.level_dimensions[level]
        mask_tumor = np.zeros((h, w)) # the init mask, and all the value is 0
    
        # get the factor of level * e.g. level 6 is 2^6
        factor = slide.level_downsamples[level]
    
        try:
            with open(json_file_path) as f:
                dicts = json.load(f)
        except Exception as e:
            print("open json file fail,ignore:{}".format(json_file_path))
            continue
        tumor_polygons = dicts['positive']
    
        for tumor_polygon in tumor_polygons:
            # plot a polygon
            name = tumor_polygon["name"]
            group_name = tumor_polygon["group_name"]
            vertices = np.array(tumor_polygon["vertices"]) / factor
            vertices = vertices.astype(np.int32)
            # different mask flag according to different group 
            code = get_label_with_group_code(group_name)["code"]
            mask_code = code
            cv2.fillPoly(mask_tumor, [vertices], (mask_code))
    
        mask_tumor = mask_tumor.astype(np.uint8)
    
        np.save(npy_file, mask_tumor)
        print("process {} ok".format(json_file))

def main():
    logging.basicConfig(level=logging.INFO)
    file_path = "/home/bavon/datasets/wsi/lsil"
    wsi_path = "{}/data".format(file_path)  
    npy_path = "{}/tumor_mask_level1".format(file_path)   
    json_path = "{}/json".format(file_path)  
    run(wsi_path,npy_path,json_path,level=1)

if __name__ == "__main__":
    main()
