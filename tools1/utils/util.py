import numpy as np
import argparse

from mmcls.apis import inference_model, init_model
from mmdet.apis import init_detector, inference_detector

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
def parse_args():
    parser = argparse.ArgumentParser(description='Test net')
    parser.add_argument('--input_folder', help='the frames path')
    parser.add_argument('--out_file', help='the dir to save results')
    args = parser.parse_args()
    if args.input_folder is None:
        #raise ValueError('--input_folder is None')
        args.input_folder="frames"
    if args.out_file is None:
        #raise ValueError('--out_file is None')  
        args.out_file="test_net.txt"
    return args

def compute_area(rec1, rec2, thr):
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        if S1 < S2:
            if S_cross/S1 > thr:
                return rec1
            else:
                return 0
        else:
            if S_cross/S2 > thr:
                return rec2 
            else:
                return 0
         
def compute_area1(rec1, rec2, thr,min_area=8000):
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
            if S_cross/S1 > thr:
                if S1>min_area:
                    return rec2
                else:
                    return rec1
            else:
                return None
        else:
            if S_cross/S2 > thr:
                if S1>min_area:
                    return rec1
                else:
                    return rec2 
            else:
                return None
            
def getscore(model, video_id, img, list_det):
    roi = img[int(list_det[3]):int(list_det[3] + list_det[5]), int(list_det[2]):int(list_det[2] + list_det[4])]
    # cv2.imwrite(os.path.join('/home/qsh/crop_test/2', 'img_%04d.jpg'%(list_det[0])), roi)
    result, scores, features = inference_model(model, roi)
    line_new = [video_id] + list(list_det[:6]) + list(scores[0])
    return line_new

def getfeature(model, img, list_det):
    
    roi = img[int(list_det[3]):int(list_det[3] + list_det[5]), int(list_det[2]):int(list_det[2] + list_det[4])]
    result, scores, features = inference_model(model, roi)
    # line_new = list(list_det[:9]) + [-1] + features[0].detach().cpu().numpy().tolist()[0]
    line_new = features[0].detach().cpu().numpy()[0]
    return line_new
def get_score_feature(model, video_id, img, list_det):
    roi = img[int(list_det[3]):int(list_det[3] + list_det[5]), int(list_det[2]):int(list_det[2] + list_det[4])]
    # cv2.imwrite(os.path.join('/home/qsh/crop_test/2', 'img_%04d.jpg'%(list_det[0])), roi)
    result, scores, features = inference_model(model, roi)
    line_new = [video_id] + list(list_det[:6]) + list(scores[0])
    f = features[0].detach().cpu().numpy()[0]
    return line_new,f

def crop(frame, tray):
    frame = frame[int(tray[1]):int(tray[3])][:][:]
    frame = np.transpose(frame, (1, 0, 2))
    frame = frame[int(tray[0]):int(tray[2])][:][:]       
    frame = np.transpose(frame, (1, 0, 2))
    
    return frame
def crop_batch(frames,tray):
    results=[]
    for frame in frames:
        frame = frame[int(tray[1]):int(tray[3])][:][:]
        frame = np.transpose(frame, (1, 0, 2))
        frame = frame[int(tray[0]):int(tray[2])][:][:]       
        frame = np.transpose(frame, (1, 0, 2))
        results.append(frame)
    return results
def montage(frame,tray,crop_frame):
    frame1=frame.copy()
    frame1[int(tray[1]):int(tray[3]),int(tray[0]):int(tray[2]),:]=crop_frame
   
    return frame1

def mask_person_box(res1,res2):
    results=[]
    for res in res1:
        a=1
        for p in res2:
            if get_iou(res,p)>0.9:
                a=0
                break
        if a==1:
            results.append(res)
    return results

class AdaptiveMaxPool1d:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, x):
        input_size = len(x)
        pool_size = input_size // self.output_size

        # 计算每个池化区域的起始和结束位置
        start_idx = np.arange(self.output_size) * pool_size
        end_idx = np.concatenate([start_idx[1:], [input_size]])

        # 在每个池化区域内找到最大值
        output = np.zeros((self.output_size,))
        for i in range(self.output_size):
            output[i] = np.max(x[start_idx[i]:end_idx[i]], axis=-1)

        return output
class AdaptiveAvgPool1d:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, x):
        input_size = len(x)
        pool_size = input_size // self.output_size

        # 计算每个池化区域的起始和结束位置
        start_idx = np.arange(self.output_size) * pool_size
        end_idx = np.concatenate([start_idx[1:], [input_size]])

        # 在每个池化区域内找到最大值
        output = np.zeros((self.output_size,))
        for i in range(self.output_size):
            output[i] = np.mean(x[start_idx[i]:end_idx[i]], axis=-1)

        return output
def feature_concat(features,out_size=512):
    res_feature=[]
    res_avg=[]
    Max_p=AdaptiveMaxPool1d(out_size)
    Avg_p=AdaptiveAvgPool1d(out_size)
    for feature in features:
        res_feature.append(Max_p(feature))
        res_avg.append(Avg_p(feature))
    res_avg=np.concatenate(res_avg,axis=0)
    res_feature=np.concatenate(res_feature,axis=0)
    res_feature=res_feature+res_avg
    return res_feature

