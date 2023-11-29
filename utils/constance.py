Label_Dict = [{"code":1,"group_code":"D","desc":"CIN 2"},
              {"code":2,"group_code":"E","desc":"CIN 3"},
              {"code":3,"group_code":"F","desc":"CIN 2 to 3"},
              ]

def get_label_with_group_code(group_code):
    for item in Label_Dict:
        if group_code==item["group_code"]:
            return item
        
def get_label_cate():
    cate = [0,1,2,3]
    # cate = [0,1]
    return cate        

def get_tumor_label_cate():
    return [1,2,3]