import numpy as np
import cv2
from mmcls.apis import inference_model, init_model
from mmdet.apis import init_detector, inference_detector
#from mmseg.apis import init_segmentor,inference_segmentor

def person_seg(model, frame):
    results = inference_detector(model, frame)
    if isinstance(results, tuple):
        bbox_result, segm_result = results
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = results, None
    if len(bbox_result[0]) == 0:
        return None
    segms = segm_result[0]
    bboxes = np.vstack(bbox_result[0])
    indices = bboxes[:, -1] > 0.1
    segms = np.stack(segms, axis=0)
    segms = segms[indices]
    if len(segms) == 0:
        return None
    return segms

def person_seg_batch(model, frames):
    result_batch=inference_detector(model,frames)
    objes=[]
    segs=[]
    for results in result_batch:
        o,s=deal_results(results)
        objes.append(o)
        segs.append(s)
    return objes,segs



def person_seg1(model, frame):
    results = inference_detector(model, frame)
    if isinstance(results, tuple):
        bbox_result, segm_result = results
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = results, None
    if len(bbox_result[0]) == 0:
        return None,None
    segms = segm_result[0]
    bboxes = np.vstack(bbox_result[0])
    indices = bboxes[:, -1] > 0.1
    segms = np.stack(segms, axis=0)
    segms = segms[indices]
    if len(segms) == 0:
        return None,None
    instances=bbox_result[0]
    obj_boxes=[np.array(instance) for instance in instances]
    return obj_boxes,segms


def deal_results(results):
    if isinstance(results, tuple):
        bbox_result, segm_result = results
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = results, None
    if len(bbox_result[0]) == 0:
        return None,None
    segms = segm_result[0]
    bboxes = np.vstack(bbox_result[0])
    indices = bboxes[:, -1] > 0.1
    segms = np.stack(segms, axis=0)
    segms = segms[indices]
    if len(segms) == 0:
        return None,None
    instances=bbox_result[0]
    obj_boxes=[np.array(instance) for instance in instances]
    return obj_boxes,segms

def maskhand(frame, tray, segms, maskframe):
    # crop segms
    # frame ：原始图片
    # tray ：托盘的位置
    # segms ：人手的mask
    # maskframe ：托盘的图片
    masks_per_frame = []
    for mask in segms:
        mask = mask[int(tray[1]):int(tray[3])][:]
        mask = np.transpose(mask, (1, 0))
        mask = mask[int(tray[0]):int(tray[2])][:]
        mask_per_frame = np.array([[0] * (int(tray[2]) - int(tray[0]))] * (int(tray[3]) - int(tray[1])))        
        mask = np.transpose(mask, (1, 0))
        masks_per_frame.append(mask)
    for mask in masks_per_frame:
        mask = np.array(mask, dtype=np.int8)
        mask_per_frame = mask_per_frame | mask
    # mask
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    # dilated = cv2.dilate(gray.copy(), kernel, 10)
    mask_part = []
    other_part = []
    first_frame_trans = np.transpose(maskframe, (2, 0, 1))
    frame = np.transpose(frame, (2, 0, 1))
    for c in range(3):
        mask_part.append(np.multiply(first_frame_trans[c][:][:], mask_per_frame))
        other_part.append(np.multiply(frame[c][:][:], ~ (mask_per_frame.astype(np.bool_))))
    frame = np.add(mask_part, other_part)
    frame = np.transpose(frame, (1, 2, 0))
    img = np.array(frame, dtype=np.uint8)
    return img

def sup_maskhand(frame,tray,segms,maskframe):
    # crop segms
    masks_per_frame = []
    for mask in segms:
        mask = mask[int(tray[1]):int(tray[3])][:]
        mask = np.transpose(mask, (1, 0))
        mask = mask[int(tray[0]):int(tray[2])][:]
        
        mask_per_frame = np.array([[0] * (int(tray[2]) - int(tray[0]))] * (int(tray[3]) - int(tray[1])))        
        mask = np.transpose(mask, (1, 0))
        masks_per_frame.append(mask)
    for mask in masks_per_frame:
        mask = np.array(mask, dtype=np.int8)
        mask_per_frame = mask_per_frame | mask
    # mask
    mask_part = []
    other_part = []
    first_frame_trans = np.transpose(maskframe, (2, 0, 1))
    frame = np.transpose(frame, (2, 0, 1))
    for c in range(3):
        mask_part.append(np.multiply(first_frame_trans[c][:][:], mask_per_frame))
        other_part.append(np.multiply(frame[c][:][:], ~ (mask_per_frame.astype(np.bool_))))
    frame = np.add(mask_part, other_part)
    frame = np.transpose(frame, (1, 2, 0))
    img = np.array(frame, dtype=np.uint8)
    # img = Image.fromarray(img)

    return img

def maskhand_mae(frame,tray,mask,maskframe):
    mask = mask[int(tray[1]):int(tray[3])][:]
    mask = np.transpose(mask, (1, 0))
    mask = mask[int(tray[0]):int(tray[2])][:]
    mask = np.transpose(mask, (1, 0))
    mask_per_frame = np.array(mask, dtype=np.int8)
    mask_part = []
    other_part = []
    first_frame_trans = np.transpose(maskframe, (2, 0, 1))
    frame = np.transpose(frame, (2, 0, 1))
    for c in range(3):
        mask_part.append(np.multiply(first_frame_trans[c][:][:], mask_per_frame))
        other_part.append(np.multiply(frame[c][:][:], ~ (mask_per_frame.astype(np.bool_))))
    frame = np.add(mask_part, other_part)
    frame = np.transpose(frame, (1, 2, 0))
    img = np.array(frame, dtype=np.uint8)
    return img

def person_seg_mae(mae_model,frame):
    pass
    # result=inference_segmentor(mae_model,frame)
    # mask=np.zeros_like(result[0])
    # mask[result[0]==12]=1
    # xs,ys=np.where(result[0]==12)
    # if len(xs)==0:
    #     return None
    # return mask