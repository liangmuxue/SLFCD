import numpy as np
import cv2

def rect_overlap(rect1,rect2):
    """取得2个矩形的重叠区域
    """
    (x11,y11,x12,y12) = rect1 
    (x21,y21,x22,y22) = rect2 
    
    # 求最小的外包矩形
    startx = min(x11,x21) 
    endx = max(x12,x22)
    starty = min(y11,y21) 
    endy = max(y12,y22)
    
    width = (x12-x11) + (x22-x21) - (endx-startx) 
    height = (y12-y11) + (y22-y21) - (endy-starty)
    
    # 取得相交的xyxy格式区域
    if width<=0 or height<=0:
        return tuple()
    else: 
        X1 = max(x11,x21) 
        Y1 = max(y11,y21)
        X2 = min(x12,x22)
        Y2 = min(y12,y22)
    # 相交的区域面积
    area = width * height        
    return (X1,Y1,X2,Y2,area)