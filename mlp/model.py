import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
class net(nn.Module):
    def __init__(self,params):
        super(net,self).__init__()
        self.input_dim=params.input_dim
        self.hid_dim=params.hid_dim
        self.num_classes=params.num_classes
        self.n_query=params.n_query
        self.batch_size=params.batch_size
        self.n_support=params.batch_size-params.n_query
        self.lin=nn.Linear(self.input_dim,self.hid_dim)
        self.bn1=nn.BatchNorm1d(self.hid_dim)
        self.relu=nn.ReLU()
        self.lin2=nn.Linear(self.hid_dim,self.hid_dim)
        self.bn2=nn.BatchNorm1d(self.hid_dim)
        self.classiffier=nn.Linear(self.hid_dim,self.num_classes)
        self.loss_fn=nn.CrossEntropyLoss()
        #self.theras=nn.Parameter(torch.tensor(1e-6))
        #self.support=torch.rand((self.num_classes,self.hid_dim)).cuda()
    def forward(self,x):
        feature=self.relu(self.bn2(self.lin2(self.relu(self.bn1(self.lin(x.float()))))))
        return self.classiffier(feature),feature
    # def set_forward(self,x):
    #     b,r,d=x.shape
    #     out=self.lin2(self.relu(self.lin(x.float())))
    #     out_=out[:,:self.n_query,:]
    #     support=out[:,self.n_query:,:]
    #     support=support.mean(1)
    #     self.support=support
    #     out_=out_.contiguous().view(self.num_classes*self.n_query,-1)
    #     return self.metric(out_,self.support)
    # def get_loss(self,x):
    #     y_query=torch.from_numpy(np.repeat(range(self.num_classes),self.n_query))
    #     y_query=Variable(y_query.cuda())
    #     y_labels=np.repeat(range(self.num_classes),self.n_query)
    #     x=x.cuda()
    #     scores=self.set_forward(x)
    #     topk_scores,topk_labels=scores.data.topk(1,1,True,True)
    #     topk_ind=topk_labels.cpu().numpy()
    #     top1_correct=np.sum(topk_ind[:,0]==y_labels)
    #     return float(top1_correct),len(y_labels),self.loss_fn(scores,y_query),scores
    # def set_forward1(self,x):
    #     out_=self.lin2(self.relu(self.lin(x.float())))
    #     out_=out_.contiguous().view(-1,self.hid_dim)
    #     return self.metric(out_,self.support)
    # def test_forward(self,x):
    #     y_query=torch.from_numpy(np.repeat(range(self.num_classes),self.batch_size))
    #     y_query=Variable(y_query.cuda())
    #     y_labels=np.repeat(range(self.num_classes),self.batch_size)
    #     x=x.cuda()
    #     scores=self.set_forward1(x)
    #     topk_scores,topk_labels=scores.data.topk(1,1,True,True)
    #     topk_ind=topk_labels.cpu().numpy()
    #     top1_correct=np.sum(topk_ind[:,0]==y_labels)
    #     return float(top1_correct),len(y_labels),self.loss_fn(scores,y_query),scores
    # def metric(self,x,y):
    #             # x: N x D
    #     # y: M x D
    #     y=y.cuda()
    #     n = x.size(0) #n_way*n_query
    #     m = y.size(0) #n_way
    #     d = x.size(1) 
    #     assert d == y.size(1)

    #     x = x.unsqueeze(1).expand(n, m, d)
    #     y = y.unsqueeze(0).expand(n, m, d)

    #     if self.n_support > 1:
    #         dist = torch.pow(x - y, 2).sum(2)
    #         score = -dist
    #     else:
    #         score = (x * y).sum(2)
    #     return score
    # def train_loop(self,dataset,opt):
    #     acc_all=[]
        
    #     for x in dataset:
    #         shape=x.shape
    #         opt.zero_grad()
    #         correct_this,count_this,loss,_=self.get_loss(x)
    #         loss.backward(retain_graph=True)
    #         opt.step()
    #         acc_all.append(correct_this/count_this*100)
    #         #counts+=1
    #     #print(counts)
    #     return np.mean(acc_all)
    
    # def test_loop(self,dataset):
    #     acc_all=[]
    #     with torch.no_grad():
    #         for x in dataset:
    #             correct_this,count_this,loss,_=self.test_forward(x)
    #             acc_all.append(correct_this/count_this*100)
    #     return np.mean(acc_all)
        