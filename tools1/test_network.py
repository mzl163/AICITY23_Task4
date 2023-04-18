from cv2 import line
from mmcls.apis import inference_model, init_model
from mmdet.apis import init_detector, inference_detector
from PIL import Image
import os
import torch
from collections import Counter
import time
import sys

from sklearn.mixture import GaussianMixture
# from .deep_sort.deep_sort_app import *
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from deep_sort.deep_sort_app import *
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np

from utils.util import  parse_args,compute_area,getfeature,getscore,crop,compute_area1,mask_person_box,get_score_feature,feature_concat
from utils.tray import find_traywohand,find_white_tray,find_wohand_Neighborhood,try_find_tray
from utils.bbox import tray_iou
from utils.deal_hand import person_seg,maskhand,sup_maskhand,person_seg1
from utils.MLP import net as mlp

mlp_model=mlp()
state_dict=torch.load("./new_checkpoints/best_DTC_single_GPU.pth")
mlp_model.load_state_dict(state_dict)
# init
# pretrain detector
pretrain_config_file = './mmdetection/configs/detectors/detectors_htc_r50_1x_coco.py'
pretrain_checkpoint_file = './new_checkpoints/detectors_htc.pth'
pretrain_detector = init_detector(pretrain_config_file, pretrain_checkpoint_file)
# detector
# config_file = 'mmdetection/configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py'
# checkpoint_file = 'checkpoints/detectors_cascade_rcnn.pth'
config_file = './mmdetection/configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py'
checkpoint_file = './new_checkpoints/detectors_cascade_rcnn.pth'
detector = init_detector(config_file, checkpoint_file)
class_num_in_detector=1

cmr_config_file = './mmdetection/work_dirs/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco.py'
cmr_checkpoint_file = './new_checkpoints/cascade_mask_rcnn.pth'
cmr_detector = init_detector(cmr_config_file, cmr_checkpoint_file)

class_num_in_detector=1
# feature model
config_file = './configs/efficientnet-b0_8xb32-01norm_in1k.py'
#checkpoint_file='/home/ldl/AiCity/DTC_AICITY2022-main/checkpoints/feature.pth'
checkpoint_file='./new_checkpoints/feature.pth'
feature_model = init_model(config_file, checkpoint_file)

# model b2
config_file = './mmclassification/configs/efficientnet/efficientnet-b2_8xb32-01norm_in1k.py'
checkpoint_file = './new_checkpoints/b2.pth'
#checkpoint_file = '/home/ldl/AiCity/DTC_AICITY2022/mmclassification-master/work_dirs/b2/epoch_1.pth'
model_b2 = init_model(config_file, checkpoint_file)
# model resnest50
config_file = './mmclassification/configs/resnest/resnest50_32xb64_in1k.py'
checkpoint_file = './new_checkpoints/s50.pth'
#checkpoint_file = '/home/ldl/AiCity/DTC_AICITY2022/mmclassification-master/work_dirs/resnest50_32xb64_in1k/epoch_3.pth'
model_s50 = init_model(config_file, checkpoint_file)
# model resnest 101
config_file = './mmclassification/configs/resnest/resnest101_32xb64_in1k.py'
checkpoint_file = './new_checkpoints/s101.pth'
model_s101 = init_model(config_file, checkpoint_file)


# 跟踪
def track(scores,video_id=0):
    # input
    # scores: [video_id, fid, score, x, y, w, h, cls_score*117, feature*1280] *n
    # output
    # scores: [video_id, fid, track_id, x, y, w, h, score*117] *n
    scores = run(
        detections=scores,
        min_confidence=0.3,
        nms_max_overlap=1.0,
        min_detection_height=0,
        max_cosine_distance=0.4, # 0.5
        nn_budget=100,
        display=False,
        video_id=video_id
    )

    return scores



def moved_process(video_id):  #托盘移动的处理方式
    # frame path
    frame_path = './frames/%d'%(video_id)
    # 找一帧白色托盘无人手
    start_time = time.time()
    white_tray = None
    
    for fid in range(1, len(os.listdir(frame_path)) + 1, 10): #100可能是隐患 因为相机的运动只需要大概三十帧 如果之捕捉到了前面的 后面可能发生变化 
        frame = os.path.join(frame_path, 'img_%04d.jpg'%(fid))
        frame = cv2.imread(frame)
        result = find_traywohand(pretrain_detector, frame)
        if result is not None:
            mask_frame = crop(frame, result)
            white_tray = result
            break
    if white_tray is None:
        for fid in range(1, len(os.listdir(frame_path)) + 1):
            frame = os.path.join(frame_path, 'img_%04d.jpg'%(fid))
            frame = cv2.imread(frame)
            result = find_traywohand(pretrain_detector, frame)
            if result is not None:
                mask_frame = crop(frame, result)
                white_tray = result
                break
    if white_tray is None:
        white_tray = [500, 250, 1370, 920]
        mask_frame = crop(cv2.imread(os.path.join(frame_path, 'img_0001.jpg')), white_tray)
    # 找不到间隔50帧再重新遍历？
    
    end_time = time.time()
    print('Find time:', end_time - start_time)

    # 分割人手 + 检测
    supervisor_frame=15 #每间隔supervisor_frame 帧去监督托盘位置是否改变
    iou_threshold=0.78 #两幅图片中IOU阈值小于设定值时 认为托盘正在发生移动
    last_white_tray=white_tray #记录上一次检查时白色托盘的位置
    moved=False #是否移动标志符
    features = []
    scores = []
    

    for fid in range(1,len(os.listdir(frame_path)) + 1):
        start_time = time.time()
        frame = os.path.join(frame_path, 'img_%04d.jpg'%(fid))
        frame = cv2.imread(frame)
        if fid%supervisor_frame==0: #进入检查点  
            white_tray1=try_find_tray(pretrain_detector,frame) #找这一帧的托盘位置
            if white_tray1 is not None:
                moved=tray_iou(white_tray,white_tray1,iou_threshold)
                if moved: #如果相对原始的位置发生移动
                    if not tray_iou(last_white_tray,white_tray1,iou_threshold): #如果它相对上一个检测点没有移动的话，认为其停止移动
                        #在其帧附近找一帧白色托盘无人手
                        inter_mask_frame,inter_white_tray=find_wohand_Neighborhood(fid,15,len(os.listdir(frame_path)) + 1,pretrain_detector,frame_path)
                        if inter_mask_frame is not None:
                            white_tray=inter_white_tray #新老交替
                            mask_frame=inter_mask_frame #新老交替
                    else: #如果相对上一个检测点托盘移动了，
                        last_white_tray=white_tray1 #更新上个监测点托盘位置，以便找出托盘停止移动的位置        
        frame_crop = crop(frame, white_tray)
        #tray_box=white_tray.copy()
        person_boxes ,result= person_seg1(pretrain_detector, frame)
        persons=[]
        if person_boxes is not None:
            for p in person_boxes:
                if p[4]>0.8:
                    persons.append(p)
        if result is not None:
            frame_crop = np.array(maskhand(frame_crop, white_tray, result, mask_frame))
        # detect
        result = inference_detector(detector, frame_crop)
        if class_num_in_detector==1:
            instances = result[0].tolist()
        else:
            instances=[]
            for re in result:
                instances.extend(re.tolist())
                
        cmr_result = inference_detector(cmr_detector, frame_crop)
        if class_num_in_detector==1:
            cmr_instances = cmr_result[0]#.tolist()
        else:
            cmr_instances=[]
            for re in cmr_result:
                cmr_instances.extend(re.tolist())
        results = []
        # 卡阈值
        cmr_instances=cmr_instances[0]
        result=mask_person_box(instances,persons)
        cmr_result=mask_person_box(cmr_instances,persons)
        instances=[np.array(instance) for instance in result]
        cmr_instances=[np.array(cmr_instance) for cmr_instance in cmr_result]
        if len(instances)==0:
                continue
        #=====================第一个检测器的结果处理============================
        results=[]
        for instance in instances:
            if instance[4] >= 0.5:
                results.append(instance.tolist())
        # 大框去小框
        rm_list = []
        for i in range(len(results) - 1):
            for j in range(i + 1, len(results)):
                rm_instance = compute_area(results[i], results[j],0.8)
                if rm_instance is not None:
                    rm_list.append(rm_instance)
        for rm_instance in rm_list:
            if rm_instance in results:
                results.remove(rm_instance)
        #====================第二个检测器的结果处理==============================
        cmr_results=[]
        for cmr_instance in cmr_instances:
            if cmr_instance[4] >= 0.4:
                cmr_results.append(cmr_instance.tolist())
        # 小框去大框
        rm_list = []
        for i in range(len(cmr_results) - 1):
            for j in range(i + 1, len(cmr_results)):
                rm_instance = compute_area1(cmr_results[i], cmr_results[j],0.8)
                if rm_instance is not None:
                    rm_list.append(rm_instance)
        for rm_instance in rm_list:
            if rm_instance in results:
                cmr_results.remove(rm_instance) 
        #两者合并      
        results.extend(cmr_results)
        #特征提取
        rect_frame1=np.ascontiguousarray(frame_crop)
        for i in results:
            #----------------------------------------------------------------------------------------------------------------------
            bbox=list(map(int,[i[0], i[1], i[2], i[3]]))
            score0,feature0 = get_score_feature(feature_model,video_id, frame_crop, [fid, -1, i[0], i[1], i[2] - i[0] + 1, i[3] - i[1] + 1, i[4], -1, -1])
            score1,feature1 = get_score_feature(model_b2, video_id, frame_crop, [fid, i[4], i[0], i[1], i[2] - i[0] + 1, i[3] - i[1] + 1, -1, -1, -1])
            score2,feature2= get_score_feature(model_s50, video_id, frame_crop, [fid, i[4], i[0], i[1], i[2] - i[0] + 1, i[3] - i[1] + 1, -1, -1, -1])
            score3,feature3 = get_score_feature(model_s101, video_id, frame_crop, [fid, i[4], i[0], i[1], i[2] - i[0] + 1, i[3] - i[1] + 1, -1, -1, -1])
            res_feature_=feature_concat([feature0,feature1,feature2,feature3])
            with torch.no_grad():
                res_feature=mlp_model.get_features(torch.from_numpy(np.array([np.concatenate([feature0,feature1,feature2,feature3])])))
            res_feature=res_feature.numpy()[0]
            #print(res_feature.shape)
            features.append(res_feature)
            scores.append(np.hstack((np.mean([score1, score2, score3,score0], axis=0), res_feature)))
            class_pre=np.mean([score1, score2, score3], axis=0)[7:]
            class_id=np.argmax(class_pre)+1
        end_time = time.time()
        print('Finished[%d/%d] cost:%.4fs/frame eta:%dm%ds'%(fid, len(os.listdir(frame_path)), end_time - start_time, \
            divmod((end_time - start_time) * (len(os.listdir(frame_path)) - fid), 60)[0], \
            divmod((end_time - start_time) * (len(os.listdir(frame_path)) - fid), 60)[1]), end='\r')
    
    results = track(scores)
    results.sort(key=lambda x: x[1])

    # 后处理
    start_time = time.time()
    instances = []
    preds = []
    for result in results:
        if [result[0], result[2]] not in instances:
            instances.append([result[0], result[2]]) #video id track id 
    for instance in instances:
        instance_frame = []
        instance_score = []
        for result in results:
            if instance == [result[0], result[2]]:
                instance_frame.append(result[1])
                instance_score.append(list(result[7:]))
        # 连续轨迹前后帧差距过大进行拆分
        idx_list = [0]
        for i in range(len(instance_frame) - 1):
            if instance_frame[i] < instance_frame[i+1] - 30: #30=====================================================
                idx_list.append(i+1)
        idx_list.append(len(instance_frame))
        for i in range(len(idx_list) - 1):
            frame_c_list = []
            frame_s_list = instance_score[idx_list[i]:idx_list[i+1]]
            for frame in frame_s_list:
                frame_c_list.append(frame.index(max(frame)) + 1)
            track_c_counter = Counter(frame_c_list)
            track_c = track_c_counter.most_common(1)
            top1_ss = []
            for frame in frame_s_list:
                top1_ss.append(frame[track_c[0][0] - 1]) #track_c[0][0] - 1 类别
            mean_score = np.mean(top1_ss)
            if len(instance_frame[idx_list[i]:idx_list[i+1]]) > 8 and track_c[0][0]< 116 and mean_score > 0.325: #5 0.25===========================================
                pred = []
                pred.append(int(instance[0]))
                pred.append(track_c[0][0])
                pred.append(int(np.mean(instance_frame[idx_list[i]:idx_list[i+1]]))) ######/60
                preds.append(pred)
            
            elif track_c[0][0]==116 and mean_score>0.7:
                pred = []
                pred.append(int(instance[0]))
                pred.append(track_c[0][0])
                pred.append(int(np.mean(instance_frame[idx_list[i]:idx_list[i+1]]))) ######/60
                preds.append(pred)
                # print("116 ",mean_score)
            # elif track_c[0][0]< 117:
            #     print(int(instance[0]),track_c[0][0],int(np.mean(instance_frame[idx_list[i]:idx_list[i+1]])),mean_score)
            #     print(len(instance_frame[idx_list[i]:idx_list[i+1]]))
    # 同类别结果取平均
    preds_idx = []
    preds_new =[]
    for pred in preds:
        if pred[:2] not in preds_idx:
            preds_idx.append(pred[:2])
            preds_new.append(pred)
        else:
            preds_new[preds_idx.index(pred[:2])].append(pred[2])
    preds_final = []
    for i in range(len(preds_new)):
        if len(preds_new[i][2:]) == 1:
            preds_final.append(preds_new[i])
            continue
        idx_list = [0]
        recover_id=[]
        for j in range(len(preds_new[i][2:])):
            if j != 0:
                if preds_new[i][2+j] - preds_new[i][1+j] > 300: #???? 6-360
                    idx_list.append(j)
                else:
                    recover_id.append(j)
        idx_list.append(len(preds_new[i][2:]))
        for j in range(len(idx_list) - 1):
            pred = []
            for id in preds_new[i][:2]:
                pred.append(id)
            mean_frame=round((max(preds_new[i][2+idx_list[j]:2+idx_list[j+1]])+min(preds_new[i][2+idx_list[j]:2+idx_list[j+1]]))/2)
            frame_ids=preds_new[i][2+idx_list[j]:2+idx_list[j+1]]
            if len(frame_ids)>1:
                mean_frame=round((max(frame_ids[1:])+min(frame_ids[1:]))/2)
                # frame_idst=[np.floor(fi/60) for fi in frame_ids]
                # min_sec,max_sec=min(frame_idst),max(frame_idst)
                # if min_sec==max_sec-1:
                #     mean_frame=int(max_sec*60-1)
            pred.append(mean_frame)
            preds_final.append(pred)
    preds = preds_final
    # 重新排序
    preds_new = []
    for i in range(5):
        tmp = []
        for pred in preds:
            if pred[0] == i + 1 :
                tmp.append(pred)
        for pred in sorted(tmp, key=lambda x:x[2]):
            preds_new.append(pred)
    end_time = time.time()
    print('Post-process cost %fs'%(end_time - start_time))

    print('Finished video%d'%(video_id))
    return preds_new

if __name__ == '__main__':
    args = parse_args()
    torch.cuda.empty_cache()
    frame_path = args.input_folder
    videos = os.listdir(frame_path)
    videos.sort(key=lambda x: int(x))
    for i in range(len(videos)):
        # break
        preds=moved_process(int(videos[i]))
        with open(args.out_file, 'a+') as f:
        # with open('./test.txt', 'a+') as f:
            for pred in preds:
                f.write(" ".join(str(i) for i in pred) + '\n')
