import os
import cv2
import numpy as np
import pandas as pd
from utils.vis import vis_data,visdom_data,repeat_with_color,ptl_to_numpy
import PIL
from openslide import ImageSlide
from torch.utils.data import DataLoader
from wsi_core.WholeSlideImage import WholeSlideImage
from clam.datasets.dataset_combine import Whole_Slide_Bag_COMBINE,Whole_Slide_Det
import h5py
import matplotlib.pyplot as plt
import openslide
from visdom import Visdom

def viz_mask(img_path,npy_path,level=1):
    # PIL.Image.MAX_IMAGE_PIXELS = 933120000
    WSI_object = WholeSlideImage(img_path)
    img = WSI_object.visWSI(vis_level=level)
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
    full_path = "/home/liang/dataset/wsi/lsil/data/1-2023_10411_01.svs"
    xml_path = "/home/liang/dataset/wsi/lsil/xml/1-2023_10411_01.xml"
    WSI_object = WholeSlideImage(full_path)
    WSI_object.initXML(xml_path)
    level = 1
    img = WSI_object.visWSI(level)
    scale = WSI_object.level_downsamples[level]
    # img.show()
    ni = np.array(img)
    vis_data(ni)
    patch_file = os.path.join("/home/liang/dataset/wsi/lsil/patches_level0/1-2023_10411_01.h5")    
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
    name = "80-CG23_15084_02"
    full_path = "/home/liang/dataset/wsi/hsil/data/{}.svs".format(name)
    xml_path = "/home/liang/dataset/wsi/hsil/xml/{}.xml".format(name)
    mask_path = "/home/liang/dataset/wsi/hsil/tumor_mask_level1/{}.npy".format(name)
    WSI_object = WholeSlideImage(full_path)
    WSI_object.initXML(xml_path)
    WSI_object.initMask(mask_path)
    level = 1
    img = WSI_object.visWSI(level)
    scale = WSI_object.level_downsamples[level]
    # img.show()
    ni = np.array(img)
    patch_file = os.path.join("/home/liang/dataset/wsi/hsil/patches_level1/{}.h5").format(name)    
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
    # plt.imshow(ni)
    # plt.axis('off')   
    img_data = ptl_to_numpy(plt) 
    # vis_data(img_data)
    visdom_data(img_data,[])

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
    file_path = "/home/liang/dataset/wsi/lsil"
    csv_path = os.path.join(file_path,"viz_data.csv")
    split_data = pd.read_csv(csv_path).values[:,0].tolist()
    wsi_path = os.path.join(file_path,"data")
    mask_path = os.path.join(file_path,"tumor_mask_level0")
    patch_level = 0
    patch_size = 512
    dataset = Whole_Slide_Det(file_path,wsi_path,mask_path,patch_path="patches_level0",patch_level=patch_level,patch_size=patch_size,split_data=split_data)
    data_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=0)   
     
    top_left = (0,0)
    
    labels = []
    theta = np.arange(0, 2*np.pi, 0.01)
    radius = 30  
    name = "1-2023_10411_01" 
    wsi_file = os.path.join(wsi_path,name + ".svs")  
    # get whole wsi data for test
    wsi = openslide.open_slide(wsi_file)      
    npy_file = os.path.join(mask_path,name+".npy") 
    region_size = wsi.level_dimensions[patch_level]
    total_img = np.array(wsi.read_region(top_left, patch_level, region_size).convert("RGB"))
    # sample_img = np.array(wsi.read_region(stf, patch_level,[256,256]).convert("RGB"))
    # sample_img = cv2.cvtColor(sample_img,cv2.COLOR_RGB2BGR) 
    # visdom_data(sample_img,[])
    mask_data = np.load(npy_file) 
    total_img = attach_mask(total_img,mask_data)  
    plt.figure(figsize=(10, 8))
    viz_tumor = Visdom(env="tumor", port=8098)
    viz_normal = Visdom(env="normal", port=8098)
    viz_number_tumor = 0
    viz_number_normal = 0
    
    for i, data in enumerate(data_loader):
        
        img_ori = data['img']
        img_ori = img_ori.cpu().numpy().squeeze(0)
        item = dataset.patches_bag_list[i]
        name = item["name"]
        coord = item["coord"]
        scale = item["scale"]
        label = item["label"]
        annot = item["bboxes"]
        # visdom_data(cv2.resize(total_img, (int(total_img.shape[1]/5),int(total_img.shape[0]/5))),[])    
        
        if label>0:   
            # 循环画出标注框   
            for anno_item in annot:
                anno_label = anno_item[-1]
                # 不同标注不同区域颜色
                if anno_label==0:
                    color_value = 0
                    color_mode = 'black'
                if anno_label==1:
                    color_value = 64    
                    color_mode = 'blue' 
                if anno_label==2:
                    color_value = 128    
                    color_mode = 'red'  
                if anno_label==3:
                    color_value = 255  
                    color_mode = 'green'             
                
                region = anno_item[:-1]
                # 相对坐标转绝对坐标
                region[0] = region[0] + coord[0]
                region[1] = region[1] + coord[1]
                region[2] = region[2] + coord[0]
                region[3] = region[3] + coord[1]
                x_min = region[0]
                x_max = region[2]
                if (x_max>total_img.shape[1]):
                    x_max = total_img.shape[1] - 1
                y_min = region[1]
                y_max = region[3]
                if (y_max>total_img.shape[0]):
                    y_max = total_img.shape[0] - 1   
                total_img[y_min:y_max,x_min:x_max,:] = color_value 
                # total_img[y_min:y_max,x_max,:] = color_value
                # total_img[y_min,x_min:x_max,:] = color_value
                # total_img[y_max,x_min:x_max,:] = color_value 
                    
                # if viz_number_tumor<10:   
                #     visdom_data(img_ori,[],viz=viz_tumor) 
                #     viz_number_tumor += 1 
                # radius = 60
        else:
            color_value = 0
            color_mode = 'black'            
            # 画出patch分割框线
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
            # 框线之间的点，醒目标记         
            radius = 30
            x = x_min + radius * np.cos(theta)
            y = y_min + radius * np.sin(theta)
            plt.fill(x, y, color=color_mode)     
            if viz_number_normal<10:   
                visdom_data(img_ori,[],viz=viz_normal) 
                viz_number_normal += 1 
    plt.imshow(total_img)
    plt.axis('off')  
    img_data = ptl_to_numpy(plt) 
    # small_img = cv2.resize(img_data, (int(img_data.shape[1]/3),int(img_data.shape[0]/3)))        
    visdom_data(img_data,[])
    # cv2.imwrite("/home/bavon/Downloads/img.png",img_data)
    
    
def viz_crop_patch(file_path,name,annotation_xywh,crop_region,patch_level=1,scale=4,viz=None):
    wsi_path = os.path.join(file_path,"data")
    mask_path = os.path.join(file_path,"tumor_mask_level{}".format(patch_level))
    wsi_file = os.path.join(wsi_path,name + ".svs")  
    # get whole wsi data for test
    wsi = openslide.open_slide(wsi_file)      
    npy_file = os.path.join(mask_path,name+".npy") 
    region_size = wsi.level_dimensions[patch_level]
    total_img = np.array(wsi.read_region((0,0), patch_level, region_size).convert("RGB"))
    total_img = cv2.cvtColor(total_img,cv2.COLOR_RGB2BGR) 
    mask_data = np.load(npy_file) 
    total_img = attach_mask(total_img,mask_data)  
    annotation_xywh = [int(annotation_xywh[0]/scale),int(annotation_xywh[1]/scale),annotation_xywh[2],annotation_xywh[3]]
    total_img = vis_data(total_img,[annotation_xywh] ,not_show=True, thickness=5)
    if crop_region is not None:
        crop_region = [crop_region[0],crop_region[2],crop_region[1],crop_region[3]]
        total_img = vis_data(total_img,[crop_region],box_mode=2,not_show=True, thickness=10)
    else:
        total_img = vis_data(total_img,[],box_mode=2,not_show=True, thickness=10)
    total_img = cv2.resize(total_img,(int(total_img.shape[1]/8),int(total_img.shape[0]/8)))
    visdom_data(total_img,[],viz=viz)
    # cv2.imwrite("/home/bavon/Downloads/img.png",img_data)
        
                         
if __name__ == '__main__':   
    img_path = "/home/bavon/datasets/wsi/test/9-CG23_12974_12.svs"
    img_path = "/home/liang/dataset/wsi/hsil/data/80-CG23_15084_02.svs"
    mask_npy_path = "/home/bavon/datasets/wsi/test/mask/9-CG23_12974_12.npy"
    mask_npy_path = "/home/liang/dataset/wsi/hsil/tumor_mask_level1/80-CG23_15084_02.npy"
    tumor_mask_npy_path = "/home/bavon/datasets/wsi/test/tumor_mask/9-CG23_12974_12.npy"
    normal_mask_npy_path = "/home/bavon/datasets/wsi/test/normal_mask/9-CG23_12974_12.npy"
    # viz_mask(img_path,mask_npy_path) 
    # viz_total()
    # viz_total_with_patch()
    # viz_total_with_masks()
    viz_within_dataset()
    
    
    