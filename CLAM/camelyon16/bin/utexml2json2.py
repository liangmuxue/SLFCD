import sys
import os
import argparse
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../')

from camelyon16.data.annotation import UteFormatter

parser = argparse.ArgumentParser(description='Convert Camelyon16 xml format to'
                                 'internal json format')


def run(args):
    
    file_path = "/home/bavon/datasets/wsi/hsil"
    # file_path = "/home/bavon/datasets/wsi/lsil"
    xml_path = os.path.join(file_path,"xml")
    json_path = os.path.join(file_path,"json")
    
    for file in os.listdir(xml_path):
        json_file = file.replace("xml", "json") 
        json_file_path = os.path.join(json_path,json_file)
        xml_file_path = os.path.join(xml_path,file)
        UteFormatter().xml2json(xml_file_path, json_file_path)

        
def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()