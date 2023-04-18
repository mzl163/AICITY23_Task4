import numpy as np
import cv2
from mmcls.apis import inference_model, init_model
from mmdet.apis import init_detector, inference_detector
import os

def crop(frame, tray):
    frame = frame[int(tray[1]):int(tray[3])][:][:]
    frame = np.transpose(frame, (1, 0, 2))
    frame = frame[int(tray[0]):int(tray[2])][:][:]       
    frame = np.transpose(frame, (1, 0, 2))

    return frame

def find_traywohand(model, frame):

    result = inference_detector(model, frame)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    if len(bbox_result[61]) == 0:  #如果找不到会报错
        return None
    if len(bbox_result[0])==0:
        return None
    # white tray
    tray_bboxes = np.vstack(bbox_result[61]) #dining table
    index_trays = np.argsort(tray_bboxes[:, -1])
    # person
    person_bboxes = np.vstack(bbox_result[0])  #persion
    person_indices = person_bboxes[:, -1] > 0.1
    person_segms = segm_result[0]
    person_segms = np.stack(person_segms, axis=0)
    person_segms = person_segms[person_indices]      # 1*1080*1920
    person_bboxes = person_bboxes[person_indices]
    # find tray without hand
    if index_trays.size:
        index_tray = index_trays[-1]
        bbox_tray = tray_bboxes[index_tray]
        if bbox_tray[-1] > 0.05: #0.5
            w = bbox_tray[2] - bbox_tray[0]
            h = bbox_tray[3] - bbox_tray[1]
            if (300 < bbox_tray[0] < 1000) and (200 < bbox_tray[1] < 700) and (800 < bbox_tray[2]) and (600 < bbox_tray[3]) and (600*500 <w*h < 900 * 700):
                for person_segm in person_segms:
                    for i in range(int(bbox_tray[0]), int(bbox_tray[2])):
                        for j in range(int(bbox_tray[1]), int(bbox_tray[3])):
                            if person_segm[j][i]:
                                return None
                return bbox_tray
    return None


def find_wohand_Neighborhood(fid,neighbor_size,length,pretrain_detector,frame_path):  #在邻域内找一帧白色托盘无人手
    for fid1 in range(max(1,fid-neighbor_size),min(fid+neighbor_size,length)):
        frame = os.path.join(frame_path, 'img_%04d.jpg'%(fid1))
        frame=cv2.imread(frame)
        result = find_traywohand(pretrain_detector, frame)
        if result is not None:
            mask_frame = crop(frame, result)
            white_tray = result
            return mask_frame,white_tray
    return None,None

def find_white_tray(start_frame_id,end_frame_id,inter_frame,pretrain_detector,frame_path):
    white_tray=None
    for fid in range(start_frame_id,end_frame_id+1,inter_frame):
        frame=os.path.join(frame_path,'img_%4d.jpg'%(fid))
        frame=cv2.imread(frame)
        result=find_traywohand(pretrain_detector,frame)
        if result is not None:
            mask_frame=crop(frame,result)
            white_tray=result
            break
    # if white_tray is None:
    #     white_tray = [500, 250, 1370, 920]
    #     mask_frame = crop(cv2.imread(os.path.join(frame_path, 'img_0001.jpg')), white_tray)
    return mask_frame,white_tray 

def try_find_tray(pretrain_detector,frame):  #在这一帧中寻找托盘（有没有人手无所谓）
    result = inference_detector(pretrain_detector, frame)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    if len(bbox_result[61]) == 0:
        return None
    # white tray
    tray_bboxes = np.vstack(bbox_result[61]) #dining table
    index_trays = np.argsort(tray_bboxes[:, -1])
    #print(tray_bboxes,index_trays)
    index_tray = index_trays[-1]
    bbox_tray = tray_bboxes[index_tray]
    w = bbox_tray[2] - bbox_tray[0]
    h = bbox_tray[3] - bbox_tray[1]
    if (300 < bbox_tray[0] < 1000) and (200 < bbox_tray[1] < 700) and (800 < bbox_tray[2]) and (600 < bbox_tray[3]) and (600*500 <w*h < 900 * 700):
        return bbox_tray
    return None