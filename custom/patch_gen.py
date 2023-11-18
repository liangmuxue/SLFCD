import sys
import os
import argparse
import logging
import time
from shutil import copyfile
from multiprocessing import Pool, Value, Lock

import openslide
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Generate patches from a given '
                                 'list of coordinates')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the input directory of WSI files')
parser.add_argument('coords_path', default=None, metavar='COORDS_PATH',
                    type=str, help='Path to the input list of coordinates')
parser.add_argument('patch_path', default=None, metavar='PATCH_PATH', type=str,
                    help='Path to the output directory of patch images')
parser.add_argument('--patch_size', default=256, type=int, help='patch size, '
                    'default 768')
parser.add_argument('--level', default=0, type=int, help='level for WSI, to '
                    'generate patches, default 0')
parser.add_argument('--num_process', default=5, type=int,
                    help='number of mutli-process, default 5')

count = Value('i', 0)
lock = Lock()


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
def process(opts):
    i, pid, x_center, y_center, args = opts
    x = int(int(x_center) - args.patch_size / 2)
    y = int(int(y_center) - args.patch_size / 2)
    wsi_path = os.path.join(args.wsi_path, pid + '.svs')
    try:
        slide = openslide.OpenSlide(wsi_path)
    except Exception as e:
        print("OpenSlide fail:,wsi_path:{}".format(wsi_path))
        return
    img = slide.read_region(
        (x, y), args.level,
        (args.patch_size, args.patch_size)).convert('RGB')
    ni = np.array(img)
    img = cv2.cvtColor(ni,cv2.COLOR_RGB2BGR)    
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    
    img.save(os.path.join(args.patch_path, str(i) + '.png'))

    global lock
    global count

    with lock:
        count.value += 1
        if (count.value) % 100 == 0:
            logging.info('{}, {} patches generated...'
                         .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                 count.value))


def run(wsi_path,coords_path,patches_path,num_process=1,patch_size=1000,level=2):
    logging.basicConfig(level=logging.INFO)

    
    if not os.path.exists(patches_path):
        os.mkdir(patches_path)

    copyfile(coords_path, os.path.join(patches_path, 'list.txt'))

    opts_list = []
    infile = open(coords_path)
    args = {"patch_size":patch_size,"patch_path":patches_path,"wsi_path":wsi_path,"level":level}
    args = Struct(**args)
    for i, line in enumerate(infile):
        if len(line)<10:
            print("line ignore:",line)
            continue
        try:
            pid, x_center, y_center = line.strip('\n').split(',')
        except Exception as e:
            print("err strip:",e)
        opts_list.append((i, pid, x_center, y_center, args))
    infile.close()

    pool = Pool(processes=num_process)
    pool.map(process, opts_list)


def main():
    # args = parser.parse_args()
    for wsi_path in ["/home/bavon/datasets/wsi/hsil","/home/bavon/datasets/wsi/lsil"]:
        for task_type in ["train","valid"]:      
            for tum_type in ["tumor","normal"]:
                txt_file = "{}/txt/{}_{}_total.txt".format(wsi_path,task_type,tum_type)
                patches_path = "{}/patches/{}_{}".format(wsi_path,tum_type,task_type)
                run(wsi_path,txt_file,patches_path)


def patch_single():
    wsi_path = "/home/bavon/datasets/wsi/test"
    txt_file = "/home/bavon/datasets/wsi/test/txt/9_tumor.txt"
    patches_path = "/home/bavon/datasets/wsi/test/patches"
    run(wsi_path,txt_file,patches_path,patch_size=200)
    
if __name__ == '__main__':
    # main()
    patch_single()
    
    
    