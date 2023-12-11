import os
from shutil import copyfile
import pandas as pd
import numpy as np
import sys
import json
import openslide
import cv2
import h5py
from utils.constance import get_combine_label_with_type,get_combine_label_dict
from utils.wsi_img_viz import viz_crop_patch

from visdom import Visdom
viz_debug = Visdom(env="debug", port=8098)

def align_xml_svs(file_path):
    """Solving the problem of inconsistent naming between SVS and XML"""
    
    wsi_path = file_path + "/data"
    ori_xml_path = file_path + "/xml_ori"
    target_xml_path = file_path + "/xml"
    
    
    for wsi_file in os.listdir(wsi_path):
        if not wsi_file.endswith(".svs"):
            continue       
        single_name = wsi_file.split(".")[0]
        if "-" in single_name and False:
            xml_single_name = single_name.split("-")[0]
        else:
            xml_single_name = single_name
        xml_single_name = xml_single_name + ".xml"
        ori_xml_file = os.path.join(ori_xml_path,xml_single_name)
        tar_xml_file = os.path.join(target_xml_path,single_name + ".xml")
        try:
            copyfile(ori_xml_file,tar_xml_file)
        except Exception as e:
            print("copyfile fail,source:{} and target:{}".format(ori_xml_file,tar_xml_file),e)
        
def build_data_csv(file_path,split_rate=0.7):
    """build train and valid list to csv"""
    
    wsi_path = file_path + "/data"
    xml_path = file_path + "/xml"
    total_file_number = len(os.listdir(xml_path))
    train_number = int(total_file_number * split_rate)
    train_file_path = file_path + "/train.csv"
    valid_file_path = file_path + "/valid.csv"
    
    list_train = []
    list_valid = []
    for i,xml_file in enumerate(os.listdir(xml_path)):
        
        single_name = xml_file.split(".")[0]
        wsi_file = single_name + ".svs"
        if i < train_number:
            list_train.append([wsi_file,1])
        else:
            list_valid.append([wsi_file,1])
    
    train_df = pd.DataFrame(np.array(list_train),columns=['slide_id','label'])
    valid_df = pd.DataFrame(np.array(list_valid),columns=['slide_id','label'])
    train_df.to_csv(train_file_path,index=False,sep=',')
    valid_df.to_csv(valid_file_path,index=False,sep=',')

def crop_with_annotation(file_path,level=1):
    """Crop image from WSI refer to annotation"""
    
    crop_img_path = file_path + "/crop_img"
    patch_path = file_path + "/patches_level{}".format(level)
    wsi_path = file_path + "/data"
    json_path = file_path + "/json"
    total_file_number = len(os.listdir(json_path))
    for i,json_file in enumerate(os.listdir(json_path)):
        json_file_path = os.path.join(json_path,json_file)  
        single_name = json_file.split(".")[0]    
        wsi_file = os.path.join(wsi_path,single_name + ".svs")  
        wsi = openslide.open_slide(wsi_file)  
        scale = wsi.level_downsamples[level]
        with open(json_file_path, 'r') as jf:
            anno_data = json.load(jf)
        # Convert irregular annotations to rectangles
        region_data = []
        label_data = []
        for i,anno_item in enumerate(anno_data["positive"]):
            vertices = np.array(anno_item["vertices"])
            group_name = anno_item["group_name"]
            label = get_label_with_group_code(group_name)['code']
            label_data.append(label)
            x_min = vertices[:,0].min()
            x_max = vertices[:,0].max()
            y_min = vertices[:,1].min()
            y_max = vertices[:,1].max()
            region_size = (int((x_max - x_min)/scale),int((y_max-y_min)/scale))
            xywh = [x_min,y_min,region_size[0],region_size[1]]
            region_data.append(xywh)
            # crop_img = np.array(wsi.read_region((x_min,y_min), level, region_size).convert("RGB"))
            # crop_img = cv2.cvtColor(crop_img,cv2.COLOR_RGB2BGR) 
            # img_file_name = "{}_{}|{}.jpg".format(single_name,i,label)
            # img_file_path = os.path.join(crop_img_path,img_file_name)
            # cv2.imwrite(img_file_path,crop_img)
            # print("save image:{}".format(img_file_name))
        # Write region data to H5
        patch_file_path = os.path.join(patch_path,single_name+".h5")  
        with h5py.File(patch_file_path, "a") as f:
            if "crop_region" in f:
                del f["crop_region"]
            f.create_dataset('crop_region', data=np.array(region_data)) 
            f['crop_region'].attrs['label_data'] = label_data
        
def patch_anno_img(xywh,patch_size=256,mask_threhold=0.9,mask_data=None,scale=4,file_path=None,label=1,file_name=None,index=0,level=1,wsi=None):
    """Crop annotation image with patch size"""
    
    tumor_patch_path = os.path.join(file_path,"tumor_patch_img")
    
    start_x,start_y,width,height = xywh
    start_x = start_x/scale
    start_y = start_y/scale
    end_x = start_x + width
    end_y = start_y + height

    
    def write_to_disk(patch_region,row=0,column=0):
        tumor_patch_file_path = os.path.join(tumor_patch_path,"{}/origin/{}_{}{}.jpg".format(label,file_name,row,column))
        top_left = (int(patch_region[0]*scale),int(patch_region[2]*scale))
        img_data = wsi.read_region(top_left, level, (patch_size, patch_size)).convert('RGB')
        img_data = cv2.cvtColor(np.array(img_data), cv2.COLOR_RGB2BGR)    
        cv2.imwrite(tumor_patch_file_path,img_data)
            
    # Ignor small image
    if width<patch_size or height<patch_size:
        return None
        # ext_w = patch_size - width
        # ext_h = patch_size - height
        # region = [int(start_x - ext_w/2),int(end_x + ext_w/2),int(start_y - ext_h/2),int(end_y + ext_h/2)]
        # write_to_disk(region)
        # return np.expand_dims(np.array(region),axis=0)
    
    def step_crop(row_index,column_index,overlap_rate=0.3):
        """Overlap crop image,Stopping crop when cross the border refer to patch length"""
        x_start = int(start_x + patch_size * column_index * overlap_rate)
        x_end = x_start + patch_size
        y_start = int(start_y + patch_size * row_index * overlap_rate)
        y_end = y_start + patch_size    
        
        if y_start>end_y:
            return None,-1         
        if x_start>end_x:
            return None,0
       
        patch_data = [x_start,x_end,y_start,y_end]
        return patch_data,0

                    
    row = 0
    patch_regions = []
    # Iterate rows and columns one by one,and crop image by patch size
    while True:
        column = 0
        while True:
            patch_region,flag = step_crop(row,column)
            # If cross the width border, then switch to next row
            if patch_region is None:
                break
            # ReFilter with mask
            patch_masked = mask_data[patch_region[2]:patch_region[3],patch_region[0]:patch_region[1]]
            if (np.sum(patch_masked>0)/(patch_size*patch_size))>mask_threhold:
                patch_regions.append(patch_region)
                # Save to disk
                write_to_disk(patch_region,row=row,column=column)
                # viz_crop_patch(file_path,file_name,xywh,patch_region,viz=viz_debug)
            # else:
            #     viz_crop_patch(file_path,file_name,xywh,patch_region)
            column += 1  
        # Cross the height border, break
        if flag==-1:
            break                 
        row += 1
    
    if len(patch_regions)>0:
        patch_regions = np.stack(patch_regions)
    else:
        patch_regions = np.array([])
    return patch_regions

def build_annotation_patches(file_path,level=1,patch_size=64):
    """Load and build positive annotation data"""
    
    patch_path = file_path + "/patches_level{}".format(level)
    wsi_path = file_path + "/data"
    for patch_file in os.listdir(patch_path):
        file_name = patch_file.split(".")[0]
        # if file_name!="9-CG23_12974_12":
        #     continue
        patch_file_path = os.path.join(patch_path,patch_file)
        wsi_file_path = os.path.join(wsi_path,file_name+".svs")
        wsi = openslide.open_slide(wsi_file_path)
        scale = wsi.level_downsamples[level]
        mask_path = os.path.join(file_path,"tumor_mask_level{}".format(level))
        npy_file = os.path.join(mask_path,file_name+".npy") 
        mask_data = np.load(npy_file)
                
        with h5py.File(patch_file_path, "a") as f:
            print("crop_region for:{}".format(patch_file_path))
            crop_region = f['crop_region'][:]
            label_data = f['crop_region'].attrs['label_data'] 
            patches = []
            patches_length = 0
            db_keys = []
            for i in range(len(label_data)):
                region = crop_region[i]
                label = label_data[i]
                # Patch for every annotation images,Build patches coordinate data list 
                patch_data = patch_anno_img(region,mask_data=mask_data,patch_size=patch_size,scale=scale,file_path=file_path,
                                            file_name=file_name,label=label,index=i,level=level,wsi=wsi)   
                if patch_data is None:
                    # viz_crop_patch(file_path,file_name,region,None)                    
                    patch_data = np.array([])
                patches_length += patch_data.shape[0]
                db_key = "anno_patches_data_{}".format(i)
                if db_key in f:
                    del f[db_key]
                f.create_dataset(db_key, data=patch_data)
                db_keys.append(db_key)
            if "annotations" in f:
                del f["annotations"]
            # annotation summarize
            f.create_dataset("annotations", data=db_keys)    
            # Record total length and label
            f["annotations"].attrs['patches_length'] = patches_length      
            f["annotations"].attrs['label_data'] = label_data        
            print("patch {} ok".format(file_name))

def aug_annotation_patches(file_path,level=1):
    import Augmentor
    tumor_patch_path = os.path.join(file_path,"tumor_patch_img")
    for label in [1,2,3]:
        img_path = os.path.join(tumor_patch_path,str(label),"origin")
        target_img_path = os.path.join(tumor_patch_path,str(label),"output")
        p = Augmentor.Pipeline(img_path,output_directory=target_img_path)
                 
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.sample(200)
        p.process()
        p.zoom_random(probability=1, percentage_area=0.8)
        p.sample(200)
        p.process()
        p.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
        p.sample(200)
        p.process()
        p.flip_left_right(probability=0.5)
        p.sample(200)
        p.process()
        p.flip_top_bottom(probability=0.5)
        p.sample(200)
        p.process()
        p.random_brightness(probability=1, min_factor=0.7, max_factor=1.2)
        p.sample(200)
        p.process()
                                    
def filter_patches_exclude_anno(file_path,level=1,patch_size=256):
    """Remove annotation patches from origin coordinates"""
    
    patch_path = file_path + "/patches_level{}".format(level)
    wsi_path = file_path + "/data"
    for patch_file in os.listdir(patch_path):
        file_name = patch_file.split(".")[0]
        patch_file_path = os.path.join(patch_path,patch_file)
        wsi_file_path = os.path.join(wsi_path,file_name+".svs")
        wsi = openslide.open_slide(wsi_file_path)
        scale = wsi.level_downsamples[level]
        mask_path = os.path.join(file_path,"tumor_mask_level{}".format(level))
        npy_file = os.path.join(mask_path,file_name+".npy") 
        mask_data = np.load(npy_file)
        
        target_coords = []  
        with h5py.File(patch_file_path, "a") as f:
            coords = f['coords'][:]
            for coord in coords:
                coord_x = int(coord[0]/scale)
                coord_y = int(coord[1]/scale)
                mask_data_item = mask_data[coord_y:coord_y+patch_size,coord_x:coord_x+patch_size]
                if np.sum(mask_data_item>0)<100:
                    target_coords.append(coord)
            attr_bak = {}
            for key in f['coords'].attrs:
                attr_bak[key] = f['coords'].attrs[key]
            del f['coords']   
            f.create_dataset('coords', data=np.array(target_coords)) 
            for key in attr_bak:
                f["coords"].attrs[key] = attr_bak[key]            
                
            print("patch {} ok".format(file_name))

def judge_patch_anno(coord,mask_data=None,scale=1,patch_size=64,thres_hold=3):
    """Judge if patch has annotation data"""
    
    coord_x = int(coord[0]/scale)
    coord_y = int(coord[1]/scale)
    mask_data_item = mask_data[coord_y:coord_y+patch_size,coord_x:coord_x+patch_size]
    # No more mask data,then not has annotation data
    if np.sum(mask_data_item>0)<thres_hold:
        return False
    return True   

def build_normal_patches_image(file_path,level=1,patch_size=64):
    """Build images of normal region in wsi"""
    
    patch_path = file_path + "/patches_level{}".format(level)
    wsi_path = file_path + "/data"
    for patch_file in os.listdir(patch_path):
        file_name = patch_file.split(".")[0]
        patch_file_path = os.path.join(patch_path,patch_file)
        wsi_file_path = os.path.join(wsi_path,file_name+".svs")
        wsi = openslide.open_slide(wsi_file_path)
        scale = wsi.level_downsamples[level]
        mask_path = os.path.join(file_path,"tumor_mask_level{}".format(level))
        npy_file = os.path.join(mask_path,file_name+".npy") 
        mask_data = np.load(npy_file)
        save_path = os.path.join(file_path,"tumor_patch_img/0",file_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        print("process file:{}".format(patch_file_path))
        with h5py.File(patch_file_path, "a") as f:
            if not "coords" in f:
                print("coords not in:{}".format(file_name))  
                continue          
            coords = f['coords'][:]
            for idx,coord in enumerate(coords):
                # Ignore annotation patches data
                if judge_patch_anno(coord,mask_data=mask_data,scale=scale,patch_size=patch_size):
                    continue
                crop_img = np.array(wsi.read_region(coord, level, (patch_size,patch_size)).convert("RGB"))
                crop_img = cv2.cvtColor(crop_img,cv2.COLOR_RGB2BGR) 
                save_file_path = os.path.join(save_path,"{}.jpg".format(idx))
                cv2.imwrite(save_file_path,crop_img)
            print("write image ok:{}".format(file_name))

def combine_mul_dataset_csv(file_path,types):
    """Combine multiple tumor type csv,To: train,valid,test"""
    
    combine_train_split = None
    combine_valid_split = None
    for type in types:
        type_csv_train = os.path.join(file_path,type,"train.csv")
        train_split = pd.read_csv(type_csv_train)
        train_split["label"] = get_combine_label_with_type(type)
        train_split.insert(train_split.shape[1], 'type', type)
        if combine_train_split is None:
            combine_train_split = train_split
        else:
            combine_train_split = pd.concat([combine_train_split,train_split])

        type_csv_valid = os.path.join(file_path,type,"valid.csv")
        valid_split = pd.read_csv(type_csv_valid)
        # Reset label value
        valid_split["label"] = get_combine_label_with_type(type)
        # Add type column
        valid_split.insert(valid_split.shape[1], 'type', type)
        if combine_valid_split is None:
            combine_valid_split = valid_split
        else:
            combine_valid_split = pd.concat([combine_valid_split,valid_split])        
    # Add patient case column
    combine_train_split.reset_index(inplace=True)
    combine_valid_split.reset_index(inplace=True)
    combine_train_split['case_id'] = combine_train_split.index
    combine_valid_split['case_id'] = combine_valid_split.index
    # split valid to valid and test
    size = combine_valid_split.shape[0]
    sp_size = int(size * 0.6)
    combine_valid_sp = combine_valid_split.iloc[:sp_size]
    combine_test_sp = combine_valid_split.iloc[sp_size:]
           
    output_path = os.path.join(file_path,"combine")
    train_file_path = os.path.join(output_path,"train.csv")
    valid_file_path = os.path.join(output_path,"valid.csv")
    test_file_path = os.path.join(output_path,"test.csv")
    combine_train_split.to_csv(train_file_path)
    combine_valid_split.to_csv(valid_file_path)
    combine_test_sp.to_csv(test_file_path)
    
                                                  
if __name__ == '__main__':   
    file_path = "/home/bavon/datasets/wsi/lsil"
    # align_xml_svs(file_path) 
    # build_data_csv(file_path)
    # crop_with_annotation(file_path)
    # build_annotation_patches(file_path)
    # aug_annotation_patches(file_path)
    # filter_patches_exclude_anno(file_path)
    # build_normal_patches_image(file_path)
    types = ["hsil","lsil"]
    combine_mul_dataset_csv("/home/bavon/datasets/wsi",types)
    