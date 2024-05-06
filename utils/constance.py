Label_Dict = [{"code":0,"group_code":"0","desc":"Normal","seq":0},
                {"code":1,"group_code":"D","desc":"CIN 2","seq":1},
              {"code":2,"group_code":"E","desc":"CIN 3","seq":2},
              {"code":3,"group_code":"F","desc":"CIN 2 to 3","seq":3},
              {"code":4,"group_code":"A","desc":"Large hollowed-out cells, transparent","seq":1},
              {"code":5,"group_code":"B","desc":"The nucleus is deeply stained, small, and heterotypic","seq":2},
              {"code":6,"group_code":"C","desc":"Small hollowed-out cells, transparent","seq":1},
              ]

Combine_Label_Dict = [
                      # {"code":1,"type":"lsil"},
                      {"code":1,"type":"hsil"},
                      {"code":0,"type":"normal"}
              ]

def get_label_with_code(code):
    for item in Label_Dict:
        if code==item["code"]:
            return item
        
def get_label_with_group_code(group_code):
    for item in Label_Dict:
        if group_code==item["group_code"]:
            return item
        
def get_label_cate(mode='hsil'):
    if mode=='hsil':
        cate = [0,1,2,3]
    elif mode=='lsil':
        cate = [0,4,5,6]
    else:
        cate = [0,1,2,3,4,5,6]        
    return cate        

def get_label_cate_num(label_code,mode='hsil'):
    item = get_label_with_code(label_code)      
    return item["seq"]

def get_tumor_label_cate(mode=1):
    if mode=='hsil':
        cate = [1,2,3]
    elif mode=='lsil':
        cate = [4,5,6]
    else:
        cate = [1,2,3,4,5,6]        
    return cate  


def get_combine_label_with_type(type):
    for item in Combine_Label_Dict:
        if type==item["type"]:
            return item["code"]
        
def get_combine_label_dict():
    dict = {}
    for item in Combine_Label_Dict:
        dict[item["type"]] = item["code"]
    return dict
