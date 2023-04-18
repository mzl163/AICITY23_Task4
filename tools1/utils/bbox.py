import numpy as np


def get_iou(tray,tray1):
    area=(tray[3]-tray[1])*(tray[2]-tray[0])
    area1=(tray1[3]-tray1[1])*(tray1[2]-tray1[0])
    left=max(tray[0],tray1[0])
    right=min(tray[2],tray1[2])
    bottom=max(tray[1],tray1[1])
    top=min(tray[3],tray1[3])
    if left>=right or top <=bottom:
        return 0
    else:
        inter=(right-left)*(top-bottom)
        iou=(inter/(area+area1-inter))*1.0
        return iou
    
def tray_iou(tray,tray1,threshold):
    area=(tray[3]-tray[1])*(tray[2]-tray[0])
    area1=(tray1[3]-tray1[1])*(tray1[2]-tray1[0])
    left=max(tray[0],tray1[0])
    right=min(tray[2],tray1[2])
    bottom=max(tray[1],tray1[1])
    top=min(tray[3],tray1[3])
    if left>=right or top <=bottom:
        return True
    else:
        inter=(right-left)*(top-bottom)
        iou=(inter/(area+area1-inter))*1.0
        if iou<threshold:
            return True
        else:
            return False
        
def merge_box(rec1,rec2,small_throld=0.9,big_throld=0.7):
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return None
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        if S1 < S2:
            if S_cross/S1 > small_throld and S_cross/S2 >big_throld:
                return rec1
            else:
                return None
        else:
            if S_cross/S2 > small_throld and S_cross/S1 >big_throld:
                return rec2
            else:
                return None
def merge_box_as_two_way(ori_boxes,cmr_boxes,thr=0.7,):
    pass