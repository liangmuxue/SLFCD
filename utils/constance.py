Label_Dict = [{"code":1,"group_code":"D","desc":"CIN 2"},
              {"code":2,"group_code":"E","desc":"CIN 3"},
              {"code":3,"group_code":"F","desc":"CIN 2 to 3"},
              {"code":4,"group_code":"A","desc":"Large hollowed-out cells, transparent"},
              {"code":5,"group_code":"B","desc":"The nucleus is deeply stained, small, and heterotypic"},
              {"code":6,"group_code":"C","desc":"Small hollowed-out cells, transparent"},
              ]

Combine_Label_Dict = [{"code":0,"type":"lsil"},
              {"code":1,"type":"hsil"}
              ]

def get_label_with_group_code(group_code):
    for item in Label_Dict:
        if group_code==item["group_code"]:
            return item
        
def get_label_cate():
    cate = [0,1,2,3,4,5,6]
    # cate = [0,1,2,3]
    return cate        

def get_tumor_label_cate():
    return [1,2,3,4,5,6]


def get_combine_label_with_type(type):
    for item in Combine_Label_Dict:
        if type==item["type"]:
            return item["code"]
        
def get_combine_label_dict():
    dict = {}
    for item in Combine_Label_Dict:
        dict[item["type"]] = item["code"]
    return dict
