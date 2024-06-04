import sys
import os
import argparse
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../')

from camelyon16.data.annotation import UteFormatter

parser = argparse.ArgumentParser(description='Convert Camelyon16 xml format to'
                                 'internal json format')

parser.add_argument('--source', type=str, default="/home/bavon/datasets/wsi/ais", 
                    help='path to folder containing raw wsi image files')

def run(args):
    
    file_path = args.source 
    # file_path = "/home/liang/datasets/wsi/lsil"
    xml_path = os.path.join(file_path,"xml")
    json_path = os.path.join(file_path,"json")
    if not os.path.exists(json_path):
        os.mkdir(json_path)
    
    for file in os.listdir(xml_path):
        json_file = file.replace("xml", "json") 
        json_file_path = os.path.join(json_path,json_file)
        xml_file_path = os.path.join(xml_path,file)
        UteFormatter().xml2json(xml_file_path, json_file_path)
        print("process successsful: ", json_file_path, '->', json_file_path)

        
def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)
    print("process successful!!!")


if __name__ == '__main__':
    main()