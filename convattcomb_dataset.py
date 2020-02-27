# -*- coding: cp936 -*-
"""
Created on Wed Jun 12 16:04:34 2019

@author: 18443
"""
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, utils
import numpy as np
from dictionary import char_index_dictionary,index_char_dictionary



class MyDataset(Dataset):
    def __init__(self,PATH1,PATH3,PATH2,transform=None):
        
        result_1=open(PATH1, 'rb').readlines()
        result_3=open(PATH3, 'rb').readlines()
        
        result_2=open(PATH2, 'rb').readlines()
        assert len(result_1)==len(result_2),'length of two outTXT must be same'
        assert len(result_1)==len(result_3),'length of two outTXT must be same'
        print(len(result_1)/3)
        print(len(result_2)/3)
        print(len(result_3)/3)
        
        self.count7356=0
        self.count11=0
        
        self.datalist=[]
        for i in range(int(len(result_1)/3)):
            
            gt=result_1[i*3+1].decode(encoding='gbk').replace('\n','')
            gtlist=[]
            gt_rawlabel=gt.split(' ')
            gt_rawlabel=self.delate7356(gt_rawlabel)
            
            if  len(gt_rawlabel)==0:
                continue
            if not self.testlabelvalidate(gt_rawlabel):
                self.count11=self.count11+1
                continue
            for k in gt_rawlabel:
                gtlist.append(int(k)+4)
            if len(gtlist)==0:
                print(i)
                print(gt)      
            gtlist.insert(0,char_index_dictionary['<s>'])
            gtlist.append(char_index_dictionary['</s>'])
            
            
            
            out1=result_1[i*3+2].decode(encoding='gbk').replace('\n','')
            indexlist1=[]
            if not out1=='':
                for k in out1.split(' '):
                    indexlist1.append(int(k)+4)
            indexlist1.insert(0,char_index_dictionary['<s>'])
            indexlist1.append(char_index_dictionary['</s>'])
            
            out3=result_3[i*3+2].decode(encoding='gbk').replace('\n','')
            indexlist3=[]
            if not out3=='':
                for k in out3.split(' '):
                    indexlist3.append(int(k)+4)
            indexlist3.insert(0,char_index_dictionary['<s>'])
            indexlist3.append(char_index_dictionary['</s>'])
            
            
            out2=result_2[i*3+1].decode(encoding='gbk').replace('\n','').replace('\r','')
            indexlist2=[]
            if not out2=='':
                for k in out2.split(','):
                    indexlist2.append(int(k)+4)
            indexlist2.insert(0,char_index_dictionary['<s>'])
            indexlist2.append(char_index_dictionary['</s>'])
            
            
            
            self.datalist.append((indexlist1,indexlist2,indexlist3,gtlist))
        print(len(self.datalist))
        print(self.count7356)
        print(self.count11)
#        def takeSecond(elem):
#            return len(elem[1])
#        #sorted_datalist=datalist
#        self.datalist.sort(key=takeSecond)
        
        self.transform = transform
            
            
    def __getitem__(self, index):
        (input1,input2,input3,label) = self.datalist[index]
        input1=torch.FloatTensor(input1)
        input2=torch.FloatTensor(input2)
        input3=torch.FloatTensor(input3)
        label=torch.FloatTensor(label)
        
        return input1,input2,input3,label

    def __len__(self):
        return len(self.datalist)
    
    def delate7356(self,s):
        news=[]
        for i in s:
            if -1<int(i)<7356:
                news.append(int(i))
        if len(news)<len(s):
            self.count7356=self.count7356+1
        return(news)
        
    def testlabelvalidate(self,s):
        count_elv=0
        count_num=0
        for i in s:
            count_num=count_num+1
            if i ==11:
                count_elv=count_elv+1
        if count_num==count_elv:
            return False
        else:
            return True


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size).fill_(char_index_dictionary['<pad>'])], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len1 = max(map(lambda x: x[0].shape[self.dim], batch))
        max_len2 = max(map(lambda x: x[1].shape[self.dim], batch))
        max_len3 = max(map(lambda x: x[2].shape[self.dim], batch))
        max_len4 = max(map(lambda x: x[3].shape[self.dim], batch))
        
#        max_len = max(max_len1,max_len2,max_len3)

        batch = list(map(lambda x:(
                pad_tensor(x[0], pad=max_len1, dim=self.dim), 
                pad_tensor(x[1], pad=max_len2, dim=self.dim),
                pad_tensor(x[2], pad=max_len3, dim=self.dim),
                pad_tensor(x[3], pad=max_len4, dim=self.dim)), batch))
        

        xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        ys = torch.stack(list(map(lambda x: x[1], batch)), dim=0)
        zs = torch.stack(list(map(lambda x: x[2], batch)), dim=0)
        ks = torch.stack(list(map(lambda x: x[3], batch)), dim=0)
#        print(xs)
#        print(ys)
        return xs, ys, zs, ks

    def __call__(self, batch):
        return self.pad_collate(batch)
PATH1="C:\\Users\\18443\\Desktop\\experiment2\\CRNN64\\testoutput\\train114wtestin114w_84accinICDAR13.txt"
#PATH1="C:\\Users\\18443\\Desktop\\experiment\\traindata\\CRNN\\train_in_109w_semantic\\removeblank\\train108wtestin108w_84accinICDAR13_remove_blank.txt"
PATH2="C:\\Users\\18443\\Desktop\\experiment\\traindata\\overseg\\all_random_result.txt"
#PATH3="C:\\Users\\18443\\Desktop\\experiment2\\EPT-recognition\\testresult\\train108wtestin108w_88accinICDAR13_attentionmodel.txt"
PATH3="C:\\Users\\18443\\Desktop\\experiment2\\EPT-recognition\\testresult\\seed1000\\train114wtestin114wRANDOM_77accinICDAR13_simpleatt_seed1000.txt"
dataset=MyDataset(PATH1,PATH3,PATH2)
train_loader = DataLoader(dataset,batch_size =1,shuffle=True, collate_fn=PadCollate(dim=0))

#testpath1="C:\\Users\\18443\\Desktop\\experiment2\\CRNN64\\testoutput\\train114wtestincompetition_84accinICDAR13.txt"
##testpath1="C:\\Users\\18443\\Desktop\\experiment\\traindata\\CRNN\\train_in_109w_semantic\\removeblank\\train108wtestincompetition_84accinICDAR13_remove_blank.txt"
#testpath2="C:\\Users\\18443\\Desktop\\experiment\\traindata\\overseg\\oversegment_testoutput_no_lm.txt"
#testpath3="C:\\Users\\18443\\Desktop\\experiment2\\EPT-recognition\\testresult\\seed1000\\train114wtestincompetition_77accinICDAR13_seed1000.txt"
#testdataset=MyDataset(testpath1,testpath3,testpath2)
#test_loader = DataLoader(testdataset,batch_size =1,shuffle=False, collate_fn=PadCollate(dim=0))

def tensor2list(tensor):
    l=[]
    for i in tensor.squeeze():
        index=int(i)
        if (index!=0)and(index!=1)and(index!=2)and(index!=3):
            l.append(index)
    return l


def tensor2string(tensor,index2word):
    string=[]
    for i in tensor.squeeze():
        index=int(i)
        if (index!=0)and(index!=1)and(index!=2)and(index!=3):
            string.append(index2word[index])
    return ''.join(string)

def editDistance(r, h):
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0: 
                d[0][j] = j
            elif j == 0: 
                d[i][0] = i
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

counter_correct_a=0
counter_correct_b=0
counter_correct_c=0
counter_number=0
for i, (batch_x, batch_y,batch_z,label) in enumerate(train_loader):
#    hanzi=[]
#    hanzi2=[]
#    hanzi3=[]
    
    l_predicta=tensor2list(batch_x)
    l_predictb=tensor2list(batch_y)
    l_predictc=tensor2list(batch_z)
    l_target=tensor2list(label)
    
    
    d1=editDistance(l_target , l_predicta)       
    counter_correct_a=counter_correct_a+d1[len(l_target)][len(l_predicta)]
    
    d2=editDistance(l_target , l_predictb)       
    counter_correct_b=counter_correct_b+d2[len(l_target)][len(l_predictb)]
    
    d3=editDistance(l_target , l_predictc)       
    counter_correct_c=counter_correct_c+d3[len(l_target)][len(l_predictc)]
    
    counter_number=counter_number+len(l_target)
    
    if i%300==0:
        print(i)
        

        print(tensor2string(batch_x,index_char_dictionary))
        print(tensor2string(batch_y,index_char_dictionary))
        print(tensor2string(batch_z,index_char_dictionary))
        print(tensor2string(label,index_char_dictionary))
        
        
        result1 = float(d1[len(l_target)][len(l_predicta)]) / len(l_target) * 100
        result1 = str("%.2f" % result1) + "%"
        
        result2 = float(d2[len(l_target)][len(l_predictb)]) / len(l_target) * 100
        result2 = str("%.2f" % result2) + "%"
        
        result3 = float(d3[len(l_target)][len(l_predictc)]) / len(l_target) * 100
        result3 = str("%.2f" % result3) + "%"
        
        print('WER1:%s'%(result1))
        print('WER2:%s'%(result2))
        print('WER3:%s'%(result3))
#
#    if i >5000:
#        break
    
        total_result1=float(counter_correct_a) / counter_number * 100
        total_result1=str("%.2f" % total_result1) + "%"
        print('test WER1:%s'%(total_result1))
        
        total_result2=float(counter_correct_b) / counter_number * 100
        total_result2=str("%.2f" % total_result2) + "%"
        print('test WER2:%s'%(total_result2))
        
        total_result3=float(counter_correct_c) / counter_number * 100
        total_result3=str("%.2f" % total_result3) + "%"
        print('test WER3:%s'%(total_result3))
        
total_result1=float(counter_correct_a) / counter_number * 100
total_result1=str("%.2f" % total_result1) + "%"
print('test WER1:%s'%(total_result1))

total_result2=float(counter_correct_b) / counter_number * 100
total_result2=str("%.2f" % total_result2) + "%"
print('test WER2:%s'%(total_result2))
        
total_result3=float(counter_correct_c) / counter_number * 100
total_result3=str("%.2f" % total_result3) + "%"
print('test WER3:%s'%(total_result3))
