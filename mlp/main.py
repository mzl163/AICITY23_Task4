import os
#os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in [0,1])

from tqdm import tqdm
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import argparse


from data_loader import BatchManager
from model import net
from dataloader import SetDataManager


def train(model,opt):
    model.train()
    train_data=[]
    test_data=[]
    for i in range(10):
        data=np.load("{}/train{}.npy".format(params.train_file,i),allow_pickle=True)
        train_data.extend([[da[1],da[0]] for da in data])
    for i in range(10):
        data=np.load("{}/val{}.npy".format(params.test_file,i),allow_pickle=True)
        test_data.extend([[da[1],da[0]] for da in data])
    trainset=BatchManager(train_data,params.batch_size,params)
    testset=BatchManager(test_data,params.batch_size,params)
    
    accs=[]
    for e in range(params.epoch):
        total_loss=0.0
        acc=0
        total=0
        for i,batch in enumerate(tqdm(trainset.iter_batch(shuffle=True))):
            opt.zero_grad()                  #?
            labels,feature = batch
            # print(len(feature),len(feature[0]))
            #print(type(feature))
            labels=torch.LongTensor(labels).cuda()
            feature=torch.FloatTensor(feature).cuda()
            #print(feature.shape[0])
            
            out,_ = model(feature)
            lost=torch.nn.CrossEntropyLoss(reduction='mean')
            loss = lost(out,labels)
            total_loss = total_loss + loss.item()
            loss.backward()
            opt.step()
        # if e<198:
        #     continue
        
        with torch.no_grad():
            for i,batch in enumerate(tqdm(testset.iter_batch())):
                labels,feature=batch
                feature=torch.FloatTensor(feature).cuda()
                out,_=model(feature)
                pred=torch.argmax(out,dim=-1).cpu().numpy()
                total+=len(labels)
                acc+=np.sum(pred==labels)
        accs.append(acc/total)
        print("{}---{}---{}".format(e+1,total_loss,acc/total))
        if max(accs)==acc/total:
            torch.save(model.state_dict(),"models/best_DTC.pth")
    print(max(accs))    
def get_num():
    num_list=[0 for _ in range(5)]     
    for i in range(5):
        a=np.load("{}/t{}.npy".format(params.test_file,i),allow_pickle=True)
        num_list[i]+=len(a)   
    print(num_list)        
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cuda',default=[0,1])
    parser.add_argument('--lr',default=0.005)
    parser.add_argument('--epoch',default=300)
    parser.add_argument('--opt',default='adam',type=str)

    parser.add_argument('--train_file',default='/home/ldl/AiCity/DTC_AICITY2022-main/mlp/data',type=str)
    parser.add_argument('--test_file',default='/home/ldl/AiCity/DTC_AICITY2022-main/mlp/data',type=str)
    
    parser.add_argument('--input_dim',default=6784)
    parser.add_argument('--hid_dim',default=2048)
    parser.add_argument('--num_classes',default=177)
    parser.add_argument('--n_query',default=15)
    parser.add_argument('--batch_size',default=320)
    parser.add_argument('--n_episode',default=500)
    parser.add_argument('--start',default=0)
    parser.add_argument('--end',default=200)
    params=parser.parse_args()
    if len(params.cuda)>0:
        os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in params.cuda)
    #params.input_dim=params.end-params.start
    model=net(params)
    
    model=torch.nn.DataParallel(model)
    cudnn.benchmark = True
    
    model=model.cuda()
    if params.opt =='sgd':
        optimzier=torch.optim.SGD(model.parameters(),lr=params.lr)
    elif params.opt=='adam':
        optimzier=torch.optim.Adam(model.parameters(),lr=params.lr)
    # train_set=SetDataManager(params,params.train_file)
    # trian_loader=train_set.get_data_loader()
    # test_set=SetDataManager(params,params.test_file)
    # test_loader=test_set.get_data_loader()
    # for e in range(params.epoch):
        
    #     acc=model.module.train_loop(trian_loader,optimzier)
    #     test_acc=model.module.test_loop(test_loader)
    #     print("epoch:{}----acc:{},test_acc:{}".format(e+1,acc,test_acc))
    #     #model.module.test_loop()
    train(model,optimzier)