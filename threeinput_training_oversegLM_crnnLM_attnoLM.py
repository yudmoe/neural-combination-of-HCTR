# -*- coding: utf-8 -*-
"""
Created on Thu May 23 20:49:32 2019

@author: 18443
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch import optim
from torch.utils.data import  DataLoader
import numpy as np
import argparse
#from convattcomb_dataset import MyDataset,PadCollate
from convattcomb_dataset import MyDataset,PadCollate
from dictionary import char_index_dictionary,index_char_dictionary

from Models.model_3single_1combineselfatt import FConvEncoder,CNN_ATT_decoder

use_cuda = torch.cuda.is_available()  # pylint: disable=no-member
device = torch.device("cuda" if use_cuda else "cpu")  # pylint: disable=no-member


parser = argparse.ArgumentParser()

parser.add_argument('--layer',
                    type=int, default=5,
                    help='layer of attention')
parser.add_argument('--PATH1',
                    default="/lustre/home/zyzhu/experiment2/traindata/CRNN/train108wtestin108w_86accinICDAR13_addLM.txt",
                    help='CRNN output txt')
parser.add_argument('--PATH2', 
                    default="/lustre/home/zyzhu/experiment/traindata/overseg/all_result_100W_with_lm.txt",
                    help='overseg output txt')
parser.add_argument('--PATH3', 
                    default="/lustre/home/zyzhu/experiment2/traindata/att/seed1006/train108wtestin108w_84accinICDAR13_seed1006.txt",
                    help='overseg output txt')

parser.add_argument('--testpath1',
                    default="/lustre/home/zyzhu/experiment2/traindata/CRNN/train108wtestincompetition_86accinICDAR13_addLM.txt",
#                    default="/lustre/home/zyzhu/experiment/traindata/CRNN/add_LM/train108wtestincompetition_84accinICDAR13_addLM.txt",
#                    default="/lustre/home/zyzhu/experiment2/CRNN64/train108wtestincompetition_88accinICDAR13_addLM.txt",
                    help='CRNN testdataset output txt')
parser.add_argument('--testpath2', 
                    default="/lustre/home/zyzhu/experiment/traindata/overseg/oversegment_testoutput_with_lm.txt",
                    help='overseg testdataset output txt')
parser.add_argument('--testpath3', 
#                    default="/lustre/home/zyzhu/experiment/traindata/CRNN/add_LM/train108wtestincompetition_84accinICDAR13_addLM.txt",
                    default="/lustre/home/zyzhu/experiment2/traindata/att/seed1006/train108wtestincompetition_84accinICDAR13_seed1006.txt",
                    help='overseg testdataset output txt')

parser.add_argument('--adam_lr', type=np.float32, default=0.00002,
                    help='learning rate')

parser.add_argument('--output_dir', default='./model_5layer_simpleattonLM_CRNNLM_overseg_combineselfatt3',
                    help='path to save model')

parser.add_argument('--batch_size', type=int, default=256,
                    help='size of one training batch')

parser.add_argument('--deviceID', type=list, default=[0,1],
                    help='deviceID')
parser.add_argument('--weight_decay', type=np.float32, default=0,
                    help='weight_decay')
parser.add_argument('--weight_clip', type=np.float32, default=0.1,
                    help='weight_decay')


opt = parser.parse_args()


    
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
   
def evaluate(encoder_a,encoder_b,encoder_c, decoder, eval_data, index2word,savepath,batch_size,epoch,printiter): 
    
    data = DataLoader(dataset=eval_data, batch_size=batch_size, collate_fn=PadCollate(dim=0))
    
    counter_correct=0
    counter_number=0
    
    for j, (batch_x, batch_y,batch_z, label) in enumerate(data):
        batch_x=batch_x.to(device).long()
        batch_y=batch_y.to(device).long()
        batch_z=batch_z.to(device).long()
        label=label.to(device).long()
        
        current_time=time.time()
        
        batch_size=batch_x.size()[0]
        
        pre_buffer=torch.zeros(batch_size,50).fill_(char_index_dictionary['<pad>'])
        pre_buffer[:,0]=char_index_dictionary['<s>']
        
        
#        preoutput_list=[char_index_dictionary['<s>']]
        
        
        encoder_a_output=encoder_a(batch_x)
        encoder_b_output=encoder_b(batch_y)
        encoder_c_output=encoder_c(batch_z)
            
            

        
        for i in range(1,50):
        
            
#            preoutput=torch.LongTensor(preoutput_list).unsqueeze(0).to(device)#list to tensor 1*length
            preoutput=pre_buffer[:,:i].long()

            output,_ =decoder(preoutput,encoder_out1=encoder_a_output,encoder_out2=encoder_b_output,encoder_out3=encoder_c_output)#B*T*7356
            
#            output,_ =decoder(preoutput,combined_output)
            
            _,prediction=torch.topk(output, 1)#B*T*1
    #        print(prediction.size())
            prediction=prediction.squeeze(2)#B*T
            
            
#            preoutput_list.append(int(prediction.squeeze(0)[-1]))
            
            if all(prediction[:,-1]==char_index_dictionary['</s>']):
                break
            
            pre_buffer[:,i]=prediction[:,-1]
            
        for one_predict_index in range(batch_size):
            
            
            l_target=tensor2list(label[one_predict_index])
            l_predict=tensor2list(pre_buffer[one_predict_index])
            
            
            d=editDistance(l_target, l_predict)
                
            counter_correct=counter_correct+d[len(l_target)][len(l_predict)]
            counter_number=counter_number+len(l_target)
                                 
                                 
        if j %printiter==0:
            print(i)
            print(j)
            
            print('time used:%s'%(time.time()- current_time))
            print(tensor2string(batch_x[one_predict_index],index_char_dictionary))
            print(tensor2string(batch_y[one_predict_index],index_char_dictionary))
            print(tensor2string(batch_z[one_predict_index],index_char_dictionary))
            print(tensor2string(label[one_predict_index],index_char_dictionary))
            print(tensor2string(prediction[one_predict_index],index_char_dictionary))
            
#                print(l_target)
#                print(l_predict)
            
            result = float(d[len(l_target)][len(l_predict)]) / len(l_target) * 100
            result = str("%.2f" % result) + "%"
            
            print('WER:%s'%(result))
        
            total_result=float(counter_correct) / counter_number * 100
            total_result=str("%.2f" % total_result) + "%"
            print(counter_correct)
            print(counter_number)
            print(' test WER of current time:%s'%(total_result))
            
    
    print(counter_correct)
    print(counter_number)
    total_result=float(counter_correct) / counter_number * 100
    total_result=str("%.2f" % total_result) + "%"
    print('test WER:%s'%(total_result))
    
    torch.save(encoder_a.state_dict(), savepath+'/encoder_a'+str(epoch)+'_acc'+str(total_result)+'.pth')
    torch.save(encoder_b.state_dict(), savepath+'/encoder_b'+str(epoch)+'_acc'+str(total_result)+'.pth')
    torch.save(encoder_c.state_dict(), savepath+'/encoder_c'+str(epoch)+'_acc'+str(total_result)+'.pth')
    torch.save(decoder.state_dict(), savepath+'/decoder'+str(epoch)+'_acc'+str(total_result)+'.pth')
    
    
#    return eval_loss.item()




def train(encoder_a,
          encoder_b,
          encoder_c,
          decoder,
          input_a,
          input_b,
          input_c,
          preout_tensor,
          target_tensor,
          encoder_a_optimizer,
          encoder_b_optimizer,
          encoder_c_optimizer,
          decoder_optimizer,
          criterion,
          weightclip
          ):

    
    
    encoder_a_optimizer.zero_grad()
    encoder_b_optimizer.zero_grad()
    encoder_c_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_a_output=encoder_a(input_a)
    encoder_b_output=encoder_b(input_b)
    encoder_c_output=encoder_c(input_c)
    
    

    output,_ =decoder(preout_tensor,encoder_out1=encoder_a_output,encoder_out2=encoder_b_output,encoder_out3=encoder_c_output)
    output=output.transpose(1, 2).contiguous()
#    print(output.size())
#    print(target_tensor.size())
    
    loss = criterion(output, target_tensor)
    
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(encoder_a.parameters(), weightclip)
    torch.nn.utils.clip_grad_norm_(encoder_b.parameters(), weightclip)
    torch.nn.utils.clip_grad_norm_(encoder_c.parameters(), weightclip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), weightclip)
    
    
    encoder_a_optimizer.step()
    encoder_b_optimizer.step()
    encoder_c_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item()


#PATH1="/lustre/home/zyzhu/CRNN64/sementic_85acc.txt"
#PATH2="/lustre/home/zyzhu/conv_att_combine/train_data/all_result_100W_no_lm.txt"
#
#testpath1="/lustre/home/zyzhu/CRNN64/competition_testoutput_85acc.txt"
#testpath2="/lustre/home/zyzhu/conv_att_combine/train_data/text_index_result_no_lm.txt"
##      
encoder_a_path="/lustre/home/zyzhu/experiment2/3+1attention/model_5layer_simpleattonLM_CRNNLM_overseg_combineselfatt3/encoder_a5_acc6.02%.pth"
encoder_b_path="/lustre/home/zyzhu/experiment2/3+1attention/model_5layer_simpleattonLM_CRNNLM_overseg_combineselfatt3/encoder_b5_acc6.02%.pth"
encoder_c_path="/lustre/home/zyzhu/experiment2/3+1attention/model_5layer_simpleattonLM_CRNNLM_overseg_combineselfatt3/encoder_c5_acc6.02%.pth"
decoder_path="/lustre/home/zyzhu/experiment2/3+1attention/model_5layer_simpleattonLM_CRNNLM_overseg_combineselfatt3/decoder5_acc6.02%.pth"

def trainIters(encoder_a,encoder_b,encoder_c, decoder, n_iters, opt):  
    
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)
        print('making folder')
        
    encoder_a.num_attention_layers = sum(layer is not None for layer in decoder.attention1)+sum(layer is not None for layer in decoder.combine_attention)
    encoder_b.num_attention_layers = sum(layer is not None for layer in decoder.attention2)+sum(layer is not None for layer in decoder.combine_attention)
    encoder_c.num_attention_layers = sum(layer is not None for layer in decoder.attention3)+sum(layer is not None for layer in decoder.combine_attention)
    
    encoder_a=torch.nn.DataParallel(encoder_a, device_ids=opt.deviceID).cuda()
    encoder_b=torch.nn.DataParallel(encoder_b, device_ids=opt.deviceID).cuda()
    encoder_c=torch.nn.DataParallel(encoder_c, device_ids=opt.deviceID).cuda()
    decoder=torch.nn.DataParallel(decoder, device_ids=opt.deviceID).cuda()
    
#    
    encoder_a.load_state_dict(torch.load(encoder_a_path))
    encoder_b.load_state_dict(torch.load(encoder_b_path))
    encoder_c.load_state_dict(torch.load(encoder_c_path))
    decoder.load_state_dict(torch.load(decoder_path))

    encoder1_optimizer = optim.Adam(encoder_a.parameters(), lr=opt.adam_lr,betas=(0.5, 0.99),weight_decay=opt.weight_decay)
    encoder2_optimizer = optim.Adam(encoder_b.parameters(), lr=opt.adam_lr,betas=(0.5, 0.99),weight_decay=opt.weight_decay)
    encoder3_optimizer = optim.Adam(encoder_c.parameters(), lr=opt.adam_lr,betas=(0.5, 0.99),weight_decay=opt.weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.adam_lr,betas=(0.5, 0.99),weight_decay=opt.weight_decay)
    

        
    
    criterion = nn.CrossEntropyLoss().to(device)
        
    dataset=MyDataset(opt.PATH1,opt.PATH3,opt.PATH2)
    
    test_dataset=MyDataset(opt.testpath1,opt.testpath3,opt.testpath2)
    print(len(test_dataset))
    
    train_loader = DataLoader(dataset,shuffle=True,batch_size =opt.batch_size, collate_fn=PadCollate(dim=0))
    
    encoder_a.eval()
    encoder_b.eval()
    encoder_c.eval()
    decoder.eval()
    with torch.no_grad():
        evaluate(encoder_a,encoder_b,encoder_c, decoder, test_dataset, index_char_dictionary,savepath=opt.output_dir,batch_size=256,epoch=0,printiter=5) 
    
    encoder_a.train()
    encoder_b.train()
    encoder_c.train()
    decoder.train()
#    
    print("start!")
    for epoch in range( n_iters ):
        #evaluate(encoder=encoder, decoder=decoder, train_data=train_data, max_length=50,index2word=index2word)
        
        for i, (batch_x, batch_y, batch_z, label) in enumerate(train_loader):
            
            
            
            batch_x=batch_x.cuda().long()
            batch_y=batch_y.cuda().long()
            batch_z=batch_z.cuda().long()
            
            label=label.cuda().long()
            
            
#            print(batch_x)
#            print(batch_y.size())
            target=label[:,1:]
            preoutput=label[:,:-1]
#            print(target)
#            print(preoutput)
            
            loss = train(encoder_a=encoder_a,encoder_b=encoder_b,encoder_c=encoder_c,
                         decoder=decoder,
                         input_a=batch_x,input_b=batch_y, input_c=batch_z, 
                         preout_tensor=preoutput,target_tensor=target, 
                         encoder_a_optimizer=encoder1_optimizer,encoder_b_optimizer=encoder2_optimizer,encoder_c_optimizer=encoder3_optimizer,
                         decoder_optimizer=decoder_optimizer, 
                         criterion=criterion,weightclip=opt.weight_clip)
            
            if i%20==0:
                print('epoch:%d,iter:%d,train_loss:%f'% (epoch,i,loss))
            
#            if (i%2000==0)and(i!=0):
        encoder_a.eval()
        encoder_b.eval()
        encoder_c.eval()
        decoder.eval()
        with torch.no_grad():
            evaluate(encoder_a,encoder_b,encoder_c, decoder, test_dataset, index_char_dictionary,savepath=opt.output_dir,batch_size=64,epoch=epoch,printiter=10) 

        encoder_a.train()
        encoder_b.train()
        encoder_c.train()
        decoder.train()


encoder_a = FConvEncoder(dictionary=char_index_dictionary,attention_layer=opt.layer)
encoder_b = FConvEncoder(dictionary=char_index_dictionary,attention_layer=opt.layer)
encoder_c = FConvEncoder(dictionary=char_index_dictionary,attention_layer=opt.layer)
decoder = CNN_ATT_decoder(dictionary=char_index_dictionary,attention_layer=opt.layer)
trainIters(encoder_a,encoder_b,encoder_c, decoder, 100,opt)      








   
        