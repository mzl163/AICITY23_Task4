import re,math,codecs,random
import numpy as np
from tqdm import trange
class BatchManager(object):
    def __init__(self,data,batch_size,args):
        self.batch_data=self.sort_and_pad(data,batch_size,args)
        self.len_data=len(self.batch_data)
        
    def sort_and_pad(self,data,batch_size,args):
        self.num_batch=int(math.ceil(len(data)/batch_size))
        print("num_batch: ",self.num_batch)
        batch_data=list()
        for i in trange(self.num_batch):
            batch_data.append(self.pad_data(data[i*batch_size:(i+1)*batch_size],args))
            
        return batch_data
    @staticmethod
    def pad_data(data,args):
        da0=[]
        da1=[]
        for t in data:
            da0.append(t[0])
            da1.append(t[1])
        return [da0,da1]
    def iter_batch(self,shuffle=True):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]
            