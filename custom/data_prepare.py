import os
from shutil import copyfile
import pandas as pd
import numpy as np
import sys

def align_xml_svs(file_path):
    """Solving the problem of inconsistent naming between SVS and XML"""
    
    wsi_path = file_path + "/data"
    ori_xml_path = file_path + "/xml_ori"
    target_xml_path = file_path + "/xml"
    
    
    for wsi_file in os.listdir(wsi_path):
        if not wsi_file.endswith(".svs"):
            continue       
        single_name = wsi_file.split(".")[0]
        if "-" in single_name:
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
           
if __name__ == '__main__':   
    file_path = "/home/bavon/datasets/wsi/hsil"
    # align_xml_svs(file_path) 
    build_data_csv(file_path)
    