import sys
import os
import argparse
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../')

from camelyon16.data.annotation import UteFormatter

parser = argparse.ArgumentParser(description='Convert Camelyon16 xml format to'
                                 'internal json format')


def run(args):
    
    hsil_xml_path = "/home/bavon/datasets/wsi/hsil/xml/"
    lsil_xml_path = "/home/bavon/datasets/wsi/lsil/xml/"
    hsil_json_path = "/home/bavon/datasets/wsi/hsil/json"
    lsil_json_path = "/home/bavon/datasets/wsi/lsil/json"
    
    
    for file in os.listdir(hsil_xml_path):
        json_file = file.replace("xml", "json") 
        json_path = hsil_json_path + "/" + json_file
        xml_path = hsil_xml_path + "/" + file  
        UteFormatter().xml2json(xml_path, json_path)
        
    for file in os.listdir(lsil_xml_path):
        json_file = file.replace("xml", "json") 
        json_path = lsil_json_path + "/" + json_file
        xml_path = lsil_xml_path + "/" + file  
        UteFormatter().xml2json(xml_path, json_path)
        
def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()