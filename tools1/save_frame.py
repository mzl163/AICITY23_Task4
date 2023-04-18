from cv2 import line
from mmcls.apis import inference_model, init_model
from mmdet.apis import init_detector, inference_detector
from PIL import Image
import os
import threading
from collections import Counter
import time
import sys
import math
import tqdm
from sklearn.mixture import GaussianMixture
# from .deep_sort.deep_sort_app import *
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from deep_sort.deep_sort_app import *
import argparse
import warnings
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

def get_feature_img(model,roi):
    result, scores, features = inference_model(model, roi)
    # line_new = list(list_det[:9]) + [-1] + features[0].detach().cpu().numpy().tolist()[0]
    line_new = features[0].detach().cpu().numpy()[0]
    return line_new

def get_four_feature(roi):
    feature0=get_feature_img(feature_model,roi)
    feature1=get_feature_img(model_b2,roi)
    feature2=get_feature_img(model_s50,roi)
    feature3=get_feature_img(model_s101,roi)
    feature=np.concatenate([feature0,feature1,feature2,feature3])
    return feature

def save_from_txt(file_dir,txt_file,i):
    with open(txt_file,"r") as f:
        txt_line=f.readlines()
    txt_line=[tl.split() for tl in txt_line]
    txt_line=[[os.path.join(file_dir,tl[0]),tl[1]] for tl in txt_line]
    data=[]
    batch=len(txt_line)//10
    if i==9:
        txt_lines=txt_line[i*batch:len(txt_line)]
    else:
        txt_lines=txt_line[i*batch:(i+1)*batch]
    for tl in tqdm.tqdm(txt_lines):
        img=cv2.imread(tl[0])
        feature=get_four_feature(img)
        label=int(tl[1])
        data.append([feature,label])
    type_=file_dir.split("/")[-1]
    np.save("./mlp/data/{}{}.npy".format(type_,i),np.array(data))
def all_(file_dir,txt_file):
    threads=[]
    for i in range(10):
        t=threading.Thread(target=save_from_txt,args=(file_dir,txt_file,i))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
all_("./data/train","./data/meta/train.txt")
all_("./data/val","./data/meta/val.txt")