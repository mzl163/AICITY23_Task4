import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import os
from abc import abstractmethod
identity=lambda x:x
class EpisodicBatchSampler(object):
    def __init__(self,n_classes,n_way,n_episodes):
        self.n_classes=n_classes
        self.n_way=n_way
        self.n_episoder=n_episodes
    def __len__(self):
        return self.n_episoder
    def __iter__(self):
        for i in range(self.n_episoder):
            yield torch.randperm(self.n_classes)[:self.n_way]
            
class SetDataset:
    def __init__(self,num_classes,batch_size,train_file,transform):
        labels=[]
        data=[]
        self.num_classes=num_classes
        self.batch_size=batch_size
        k=0
        for i in range(self.num_classes):
            ds=np.load("{}/no{}.npy".format(train_file,i))
            data.extend(ds)
            labels.extend([i for _ in range(len(ds))])
        self.data=data
        self.labels=labels
        self.transform=transform
        self.cl_list=np.unique(self.labels).tolist()
        
        self.sub_meta={}
        for cl in self.cl_list:
            self.sub_meta[cl]=[]
        
        for x,y in zip(self.data,self.labels):
            self.sub_meta[y].append(x) #按类分 
        
        self.sub_dataloader=[]
        sub_data_loader_params=dict(batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=False)
        
        for cl in self.cl_list:
            sub_dataset=SubDataset(self.sub_meta[cl],cl,transform=transform) #每一个类生成一个dataset
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset,**sub_data_loader_params)) #很关键
    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i])) #很关键 保证一个batch每个类都有 不会重复 
    def __len__(self):
        return len(self.cl_list)
    
class SubDataset:
    def __init__(self,data,cl,transform=transforms.ToTensor(),target_transform=identity):
        self.data=data
        self.cl=cl
        self.transform=transform
        self.target_transform=target_transform
    def __getitem__(self,i):
        data=self.data[i]
        #target=self.target_transform(self.cl)
        return data
    def __len__(self):
        return len(self.data)
    
            
class DataManager:
    @abstractmethod
    def get_data_loader(self,data_file,aug):
        pass
class SetDataManager(DataManager):
    def __init__(self,params,file):
        super(SetDataManager,self).__init__()
        self.num_classes=params.num_classes
        self.n_query=params.n_query
        self.batch_size=params.batch_size
        self.n_episode=params.n_episode
        self.train_file=file
        
    def get_data_loader(self):
        
        dataset=SetDataset(self.num_classes,self.batch_size,self.train_file,transform=transforms.ToTensor())
        sampler=EpisodicBatchSampler(len(dataset),self.num_classes,self.n_episode)  # 片段
        #print(self.batch_size)
        data_loader_params=dict(
            batch_sampler=sampler,  #片段
            # batch_size=self.batch_size,
            # shuffle=True,
            num_workers=0, #同上
            pin_memory=True
        )
        data_loader=torch.utils.data.DataLoader(dataset,**data_loader_params)
        return data_loader
    