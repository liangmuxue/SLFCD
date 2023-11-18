import os
import cv2
import numpy as np
import pandas as pd
from utils.vis import vis_data,visdom_data,repeat_with_color,ptl_to_numpy
import PIL
from openslide import ImageSlide
from torch.utils.data import DataLoader
from wsi_core.WholeSlideImage import WholeSlideImage
from clam.datasets.dataset_combine import Whole_Slide_Bag_COMBINE
import h5py
import matplotlib.pyplot as plt

def viz_mask(img_path,npy_path):
    # PIL.Image.MAX_IMAGE_PIXELS = 933120000
    WSI_object = WholeSlideImage(img_path)
    img = WSI_object.visWSI(vis_level=2)
    ni = np.array(img)
    img = cv2.cvtColor(ni,cv2.COLOR_RGB2BGR) 
    mask_data = np.load(npy_path)
    mask_data = np.stack([mask_data for i in range(3)])
    mask_data = mask_data.transpose(2,1,0) + 0
    ni = ni * mask_data
    ni = ni.astype(np.uint8)
    visdom_data(ni,[])

def viz_total():
    full_path = "/home/bavon/datasets/wsi/test/data/9.svs"
    xml_path = "/home/bavon/datasets/wsi/test/xml/9.xml"
    WSI_object = WholeSlideImage(full_path)
    WSI_object.initXML(xml_path)
    img = WSI_object.visWSI(0)
    # img.show()
    ni = np.array(img)
    vis_data(ni,[])
    
def viz_total_with_patch():
    full_path = "/home/bavon/datasets/wsi/test/data/12.svs"
    xml_path = "/home/bavon/datasets/wsi/test/xml/12.xml"
    WSI_object = WholeSlideImage(full_path)
    WSI_object.initXML(xml_path)
    level = 1
    img = WSI_object.visWSI(level)
    scale = WSI_object.level_downsamples[level]
    # img.show()
    ni = np.array(img)
    vis_data(ni)
    patch_file = os.path.join("/home/bavon/datasets/wsi/test/patches/12.h5")    
    theta = np.arange(0, 2*np.pi, 0.01)
    radius = 30
    plt.imshow(ni)
    with h5py.File(patch_file, "r") as f:
        coords = np.array(f['coords']) / scale
        for coord in coords:
            x = coord[0] + radius * np.cos(theta)
            y = coord[1] + radius * np.sin(theta)
            plt.fill(x, y, 'r')
    plt.axis('off')   
    img_data = ptl_to_numpy(plt) 
    vis_data(img_data)

def viz_total_with_masks():
    name = "12"
    full_path = "/home/bavon/datasets/wsi/test/data/{}.svs".format(name)
    xml_path = "/home/bavon/datasets/wsi/test/xml/{}.xml".format(name)
    mask_path = "/home/bavon/datasets/wsi/test/tumor_mask/{}.npy".format(name)
    WSI_object = WholeSlideImage(full_path)
    WSI_object.initXML(xml_path)
    WSI_object.initMask(mask_path)
    level = 1
    img = WSI_object.visWSI(level)
    scale = WSI_object.level_downsamples[level]
    # img.show()
    ni = np.array(img)
    patch_file = os.path.join("/home/bavon/datasets/wsi/test/patches/{}.h5").format(name)    
    theta = np.arange(0, 2*np.pi, 0.01)
    radius = 30
    mask_data = WSI_object.mask_data
    ni = attach_mask(ni,mask_data)
    # vis_data(ni,[])  
    with h5py.File(patch_file, "r") as f:
        coords = np.array(f['coords']) / scale
        for coord in coords:
            x = coord[0] + radius * np.cos(theta)
            y = coord[1] + radius * np.sin(theta)
            plt.fill(x, y, color='black')
    plt.imshow(ni)
    plt.axis('off')   
    img_data = ptl_to_numpy(plt) 
    vis_data(img_data)

def attach_mask(img_data,mask_data):
    color_mask_data1 = repeat_with_color([128,0,0],img_data.shape[:2])
    color_mask_data2 = repeat_with_color([0,128,0],img_data.shape[:2])
    color_mask_data3 = repeat_with_color([0,0,128],img_data.shape[:2])   
    
    idx = (mask_data==1)
    img_data[idx] = color_mask_data1[idx]
    idx = (mask_data==2)
    img_data[idx] = color_mask_data2[idx]
    idx = (mask_data==3)
    img_data[idx] = color_mask_data3[idx] 
    return img_data
         
    
def viz_within_dataset():
    file_path = "/home/bavon/datasets/wsi/test"
    csv_path = os.path.join(file_path,"valid.csv")
    split_data = pd.read_csv(csv_path).values[:,0].tolist()
    wsi_path = os.path.join(file_path,"data")
    mask_path = os.path.join(file_path,"tumor_mask")
    patch_level = 0
    dataset = Whole_Slide_Bag_COMBINE(file_path,wsi_path,mask_path,patch_level=patch_level,split_data=split_data)
    data_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=0)   
     
    top_left = (0,0)
    name = "12"
    wsi = dataset.wsi_data[name]
    patch_size = 256
    region_size = wsi.level_dimensions[patch_level]
    total_img = np.array(wsi.read_region(top_left, patch_level, region_size).convert("RGB"))
    # total_img = cv2.cvtColor(total_img, cv2.COLOR_RGB2BGR)
    total_img = attach_mask(total_img,dataset.mask_data[name])
    # visdom_data(cv2.resize(total_img, (int(total_img.shape[1]/5),int(total_img.shape[0]/5))),[])
    labels = []
    scale = wsi.level_downsamples[patch_level]
    corrds = np.array([item['coord'] for item in dataset.patches_bag_list])
    theta = np.arange(0, 2*np.pi, 0.01)
    radius = 30    
    plt.figure(figsize=(10, 8))
    for i, data in enumerate(data_loader):
        (img,label) = data
        label = label.item()
        labels.append(label)
        if label==0:
            color_value = 0
            color_mode = 'black'
        if label==1:
            color_value = 64    
            color_mode = 'blue' 
        if label==2:
            color_value = 128    
            color_mode = 'red'  
        if label==3:
            color_value = 255  
            color_mode = 'green'                                      
        item = dataset.patches_bag_list[i]
        coord = (item['coord']).astype(np.int16)
        x_min = coord[0]
        x_max = coord[0] + patch_size
        if (x_max>total_img.shape[1]):
            x_max = total_img.shape[1]- 1
        y_min = coord[1]
        y_max = coord[1] + patch_size
        if (y_max>total_img.shape[0]):
            y_max = total_img.shape[0] - 1            
        total_img[y_min:y_max,x_min,:] = color_value 
        total_img[y_min:y_max,x_max,:] = color_value
        total_img[y_min,x_min:x_max,:] = color_value
        total_img[y_max,x_min:x_max,:] = color_value
        
        # print("draw point,x:{},y:{}".format(x_min,y_min))
        x = x_min + radius * np.cos(theta)
        y = y_min + radius * np.sin(theta)
        plt.fill(x, y, color=color_mode)        

    plt.imshow(total_img)
    plt.axis('off')  
    img_data = ptl_to_numpy(plt) 
    # small_img = cv2.resize(img_data, (int(img_data.shape[1]/3),int(img_data.shape[0]/3)))        
    visdom_data(img_data,[])         
    labels = np.array(labels)
    total_len = labels.shape[0]
    mask_len = np.sum(labels>0)
    print("total_len:{},mask_len:{}".format(total_len,mask_len))
                     
if __name__ == '__main__':   
    img_path = "/home/bavon/datasets/wsi/test/9-CG23_12974_12.svs"
    mask_npy_path = "/home/bavon/datasets/wsi/test/mask/9-CG23_12974_12.npy"
    tumor_mask_npy_path = "/home/bavon/datasets/wsi/test/tumor_mask/9-CG23_12974_12.npy"
    normal_mask_npy_path = "/home/bavon/datasets/wsi/test/normal_mask/9-CG23_12974_12.npy"
    # viz_mask(img_path,normal_mask_npy_path) 
    # viz_total()
    # viz_total_with_patch()
    # viz_total_with_masks()
    viz_within_dataset()
    
    
    