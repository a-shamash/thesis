#!/usr/bin/env python

#Generator Model in Pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from numpy.random import seed, shuffle

###############################################################################
#Hyper Parameter List
########################################
#net_name: name of directory to save
#batch_size: batch size train for gen; batch size real+batch size fake for dis
#epoch_num: number of epochs
#epoch_on: start training dis on epoch_on
#dis_lim: if dis losses higher, dont train dis
#k_steps: dis training steps
#
#lr_gen: Generators Learning Rates
#lr_dis: Disriminators Learning Rates 
#
#alpha: WC vs HR loss weight
#beta: MSE vs Dis on Dec loss weight
#gamma: MSE vs DIS on Rec loss weight
#
#initialize: initialize Rec or not
#
#rec_hold: start training rec on epoch rec_hold
#validate: statistics every 'validate' steps
#
#load: Load previously trained epoch  
#load_epoch: Which epoch to load (if load=False, set load_epoch = -1)
###############################################################################

net_name = 'Network01/'
batch_size = 16 
epoch_num = 8 
epoch_on = 2 
dis_lim = 5. 
k_steps = 3

lr_gen = 0.0001 
lr_dis = 0.001

alpha = 0.75 
beta = 0.25 
gamma = 0.5 

initialize = True 
rec_hold = 5 

validate = 100

load = True
load_epoch = 5


print('Parameter List for ' + net_name)
print('\tbatch_size: ', batch_size)
print('\tepoch_num: ', epoch_num)
print('\tepoch_on: ', epoch_on)
print('\tdis_lim: ', dis_lim)
print('\tk_steps: ', k_steps)
print('\n')
print('\tlr_gen: ', lr_gen)
print('\tlr_dis: ', lr_dis)
print('\n')
print('\talpha: ', alpha) 
print('\tbeta: ', beta) 
print('\tgamma: ', gamma)
print('\n')
print('\tinitialize: ', initialize) 
print('\trec_hold: ', rec_hold) 
print('-'*30)

DATA_PATH = '/staging/leuven/stg_00032/SR/Data/'
RESU_PATH = '/staging/leuven/stg_00032/SR/Results/'+net_name



#Data Loading
class ImageDataset(Dataset):
    def __init__(self, train_data_lr, train_data_wc, train_data_hr, train = 'train', div = (.8, .9)):
            """
            Args:
                train_data_lr (string): Path to the LR data
                train_data_wc (string): Path to the WC data
                train_data_hr (string): Path to the HR data
                train: training phase: 'train', 'cross', 'test', 'all'
                div: division of samples
            """
            self.train_data_lr = np.load(train_data_lr, mmap_mode= 'r')
            self.train_data_wc = np.load(train_data_wc, mmap_mode= 'r')
            self.train_data_hr = np.load(train_data_hr, mmap_mode= 'r')
            total = self.train_data_lr.shape[0]

            seed(100)
            index = np.arange(total)
            shuffle(index)

            if train == 'train':
                self.index = index[:int(div[0]*total)]	
            elif train == 'val':
                self.index = index[int(div[0]*total):int(div[1]*total)]
            elif train == 'test':
                self.index = index[int(div[1]*total):]	
            else:
                self.index = index	


    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        i = self.index[idx]
        sample = {'LR': torch.from_numpy(self.train_data_lr[i,:,:]),
                  'WC': torch.from_numpy(self.train_data_wc[i,:,:,:]),
                  'HR': torch.from_numpy(self.train_data_hr[i,:,:,:])}
        return sample



images = ImageDataset(DATA_PATH+'train_data_lr.npy',
                      DATA_PATH+'train_data_wc.npy', 
                      DATA_PATH+'train_data_hr.npy')


print('Images: ', len(images))
dataloader = DataLoader(images, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

discriminator_loader = DataLoader(images, batch_size=batch_size*2,
                                         shuffle=True, num_workers=2)


print('Data Loaded')
print('-'*30)


#Create models
#Decomposer
class Decomposition(nn.Module):
    def __init__(self, height, width, channels=1):
        super(Decomposition, self).__init__()
        self.conv1 = nn.Conv2d(channels,64, 5)
        
        #Residual
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        
        #Residual
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        
        self.conv6 = nn.Conv2d(64, 32, 3)
        self.conv7 = nn.Conv2d(32,  3, 3)
        self.conv8 = nn.Conv2d( 3,  3, 3)


    def forward(self, x):
        conv1 = nn.ReplicationPad2d(2)(x)
        conv1 = F.relu(self.conv1(conv1))
        
        conv2 = nn.ReplicationPad2d(1)(conv1)
        conv2 = F.relu(self.conv2(conv2))
        conv3 = nn.ReplicationPad2d(1)(conv2)
        conv3 = self.conv3(conv3)
        resid1 = conv1+conv3
        
        conv4 = nn.ReplicationPad2d(1)(resid1)
        conv4 = F.relu(self.conv4(conv4))
        conv5 = nn.ReplicationPad2d(1)(conv4)
        conv5 = self.conv5(conv5)
        resid2 = resid1+conv5

        conv6 = nn.ReplicationPad2d(1)(resid2)
        conv6 = F.relu(self.conv6(conv6))

        conv7 = nn.ReplicationPad2d(1)(conv6)
        conv7 = self.conv7(conv7)
        conv8 = nn.ReplicationPad2d(1)(conv7)
        conv8 = self.conv8(conv8)
        
        return conv8

class Discriminator(nn.Module):
    def __init__(self, height, width, channels=3):
        super(Discriminator, self).__init__()
        self.height = height
        self.width = width
        self.conv1 = nn.Conv2d(channels,64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3, stride = 2)

        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64,  3, 3)
        self.dens1 = nn.Linear(int(self.height*self.width*3/4), self.height)
        self.dens2 = nn.Linear(self.height, 1) 

    def forward(self, x):
        conv1 = nn.ReplicationPad2d(1)(x)
        conv1 = F.leaky_relu(self.conv1(conv1))
        
        conv2 = nn.ReplicationPad2d(1)(conv1)
        conv2 = F.leaky_relu(self.conv2(conv2))
       
        conv3 = nn.ReplicationPad2d(1)(conv2)
        conv3 = self.conv3(conv3)
        
        conv4 = nn.ReplicationPad2d(1)(conv3)
        conv4 = F.leaky_relu(self.conv4(conv4))

        flat = conv4.view(-1, int(self.height*self.width*3/4))
        dens1 = F.leaky_relu(self.dens1(flat))
        logits = self.dens2(dens1)        
        logits = torch.squeeze(logits) 

        return logits


def combine(x, y, axis = 0):
    """combines x and y for reconstrution"""
    shape = x.shape[-1]
    N = x.shape[0]
    if axis == 0:
        z= torch.ones(N, 2*shape, shape)
        z=Variable(z).cuda()
        z[:, 0::2, :] = x
        z[:, 1::2, :] = y
    else:
        z= torch.ones(N, 2*shape, 2*shape)
        z=Variable(z).cuda()
        z[:, :, 0::2] = x
        z[:, :, 1::2] = y
    return z


#recreater
class Reconstruction(nn.Module):
    def __init__(self, height, width):
        super(Reconstruction, self).__init__()
        self.conv1 = nn.Conv3d(1, 2, (2,1,1), stride = (2,1,1), bias = False)
        self.conv2 = nn.Conv3d(1, 2, (2,1,1), stride = 1, bias = False)
        if initialize:
            self.init()

    def init(self):
        nn.init.constant(self.conv1.weight, 0.7071067811865476)    
        self.conv1.weight.data[1,0,1,0,0] = -0.7071067811865476
        nn.init.constant(self.conv2.weight, 0.7071067811865476)    
        self.conv2.weight.data[1,0,1,0,0] = -0.7071067811865476
 

    def forward(self, x):
        #x =(N, C= 4, H, W)
        x=x.unsqueeze(1)
        #x = (N,C =1, D=4, H, W)
        conv1 = self.conv1(x)
        #conv1 = (N, C=2, D=2, H, W)
        l = conv1[:,:, 0, :,:]
        h = conv1[:,:, 1, :,:]
        #l = (N, C=2, H, W)
        l = combine(l[:, 0, :, :], l[:, 1, :, :], 0)
        h = combine(h[:, 0, :, :], h[:, 1, :, :], 0)
        #l = (N, 2*H, W)
        l= l.unsqueeze(1)
        h= h.unsqueeze(1)
        
        l= l.unsqueeze(1)
        h= h.unsqueeze(1)

        mid = torch.cat((l, h), 2)
        #mid = (N,C= 1, D=2, 2*H, W)
        conv2 = self.conv2(mid)
        #conv2 = (N, C=2, D=1, 2*H, W)  
        result = combine(conv2[:,0, 0, :, :], conv2[:,1, 0, :, :], 1)
        #result = (N, 2*H, 2*W)
        result = result.unsqueeze(1)
        #result = (N,C= 1, 2*H, 2*W)
        return result

def Preformance(Rec):
    params = list(Rec.parameters())
    low = np.array([0.7071067811865476,0.7071067811865476])
    high = np.array([0.7071067811865476,-0.7071067811865476])
    

    total = 0
    #rows vs columns wavelet recomposition
    for i in range(2):
        #l* vs h*
        total += abs(params[i][0, 0, 0, 0, 0].data.cpu().numpy()-low[0])
        total += abs(params[i][0, 0, 1, 0, 0].data.cpu().numpy()-low[1])
        total += abs(params[i][1, 0, 0, 0, 0].data.cpu().numpy()-high[0])
        total += abs(params[i][1, 0, 1, 0, 0].data.cpu().numpy()-high[1])

    denom = np.sum(np.abs(low)) + np.sum(np.abs(high))
    denom *= 2
    return float(total/denom)



def PreformanceSqrt(Rec):
    params = list(Rec.parameters())
    low = np.array([0.7071067811865476,0.7071067811865476])
    high = np.array([0.7071067811865476,-0.7071067811865476])
    

    total = 0
    #rows vs columns wavelet recomposition
    for i in range(2):
        #l* vs h*
        total += (params[i][0, 0, 0, 0, 0].data.cpu().numpy()-low[0])**2
        total += (params[i][0, 0, 1, 0, 0].data.cpu().numpy()-low[1])**2
        total += (params[i][1, 0, 0, 0, 0].data.cpu().numpy()-high[0])**2
        total += (params[i][1, 0, 1, 0, 0].data.cpu().numpy()-high[1])**2

    denom = np.sum((low)**2) + np.sum((high)**2)
    denom *= 2
    return float(total/denom)

device = torch.cuda.device("cuda")


Dec = Decomposition(64,64)
Rec = Reconstruction(64,64)

decDis = Discriminator(64,64)
recDis = Discriminator(64*2,64*2, channels = 1)




if torch.cuda.device_count() > 1:
    print('Number of GPUs ', torch.cuda.device_count())
    Dec = nn.DataParallel(Dec)
    Rec = nn.DataParallel(Rec)
    decDis = nn.DataParallel(decDis)
    recDis = nn.DataParallel(recDis)
Dec.cuda()
Rec.cuda()
decDis.cuda()
recDis.cuda()


total_loss_train = []
loss_comp_train = []
preformance = []
#load models
if load:
    Dec.load_state_dict(torch.load(RESU_PATH+'GAN_Dec'+str(load_epoch)))
    Rec.load_state_dict(torch.load(RESU_PATH+'GAN_Rec'+str(load_epoch)))
    decDis.load_state_dict(torch.load(RESU_PATH+'GAN_decDis'+str(load_epoch)))
    recDis.load_state_dict(torch.load(RESU_PATH+'GAN_recDis'+str(load_epoch)))
    
    total_loss_train = list(np.load(RESU_PATH +'total_loss_train.npy'))[:23*(1+load_epoch)] 
    loss_comp_train = list(np.load(RESU_PATH +'loss_comp_train.npy'))[:23*(1+load_epoch)] 
    preformance = list(np.load(RESU_PATH +'preformance.npy'))[:23*(1+load_epoch)] 

#Set Loss
MSE = nn.MSELoss()
criter_discrim = nn.BCEWithLogitsLoss()

gen_group = [{'params': Dec.parameters()}, {'params': Rec.parameters()}]
#optimizer = optim.SGD(gen_group, lr=lr_gen, momentum=0.9)
#optimizer_hold = optim.SGD(Dec.parameters(), lr=lr_gen, momentum=0.9)


optimizer = optim.Adam(gen_group, lr = lr_gen)
optimizer_hold = optim.Adam(Dec.parameters(), lr = lr_gen)



BCE = nn.BCEWithLogitsLoss()
optimizer_decDis = optim.SGD(decDis.parameters(), lr=lr_dis, momentum=0.9)
optimizer_recDis = optim.SGD(recDis.parameters(), lr=lr_dis, momentum=0.9)


#labels for training discriminators
dis_labels = torch.ones(batch_size*2)
dis_labels[:batch_size] = 0
dis_labels = Variable(dis_labels.float()).cuda()

loss_labels = Variable(torch.ones(batch_size)).cuda()




print('Models Made')
print('-'*30)


for epoch in range(load_epoch+1, epoch_num):  
    print('\tEpoch #', epoch)
    for i, data in enumerate(dataloader, 0):
        inputs = Variable(data['LR'].float()).cuda() #.to(device)
        labels_wc = Variable(data['WC'].float()).cuda() #.to(device)
        labels_hr = Variable(data['HR'].float()).cuda() #.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        wavelets = Dec(inputs)
        
        loss_mse_wc = MSE(wavelets, labels_wc)*100
        
        logits_wc = decDis(wavelets)
        loss_bce_wc = BCE(logits_wc, loss_labels)

        wavelets_combine = torch.cat((inputs*2, wavelets), 1)
        outputs = Rec(wavelets_combine)
        loss_mse_hr = MSE(outputs, labels_hr)*100
        logits_hr = recDis(outputs)
        loss_bce_hr = BCE(logits_hr, loss_labels)
        
        if epoch < epoch_on:
            loss_hr = loss_mse_hr
            loss_wc = loss_mse_wc
        else:
            loss_hr = gamma*loss_mse_hr + (1-gamma)*loss_bce_hr
            loss_wc = beta*loss_mse_wc+(1-beta)*loss_bce_wc

        loss = alpha*loss_wc+(1-alpha)*loss_hr
        loss.backward()
        
        if epoch < rec_hold:
            optimizer_hold.step()
        else:
            optimizer.step()

        #save statistics           
        if i%validate == 0:
            """
            print('Losses Step #', i)
            print('\tMSE_WC: ', float(loss_mse_wc))
            print('\tDis_WC: ', float(loss_bce_wc))
            print('\tMSE_HR: ', float(loss_mse_hr))
            print('\tDis_HR: ', float(loss_bce_hr))
            """
            #Training
            loss_comp_train.append([float(loss_mse_wc), 
                                    float(loss_bce_wc), 
                                    float(loss_mse_hr), 
                                    float(loss_bce_hr)])
            total_loss_train.append(float(loss))           
            preformance.append(Preformance(Rec))

            np.save(RESU_PATH+'loss_comp_train.npy', np.array(loss_comp_train))
            np.save(RESU_PATH+'total_loss_train.npy', np.array(total_loss_train))
            np.save(RESU_PATH+'preformance.npy', np.array(preformance))

        
        if float(loss_bce_wc+loss_bce_hr) < dis_lim and epoch >= epoch_on:     
        #train discriminators seperatly
            print('Dis Step #', i)
            for k, dis_data in enumerate(discriminator_loader,0):
                optimizer_decDis.zero_grad()
                optimizer_recDis.zero_grad()

                inputs_dis = Variable(dis_data['LR'].float()).cuda()
                labels_wc_dis = Variable(dis_data['WC'].float()).cuda()
                labels_hr_dis = Variable(dis_data['HR'].float()).cuda()
                
                #train decDis
                fake_start = inputs_dis[:batch_size, :, :, :]
                fake = Dec(fake_start)
                real = labels_wc_dis[batch_size:, :, :, :]
                wavelet_test = torch.cat((fake, real), 0)
                
                logit_decDis = decDis(wavelet_test)
                loss_decDis = BCE(logit_decDis, dis_labels)
                loss_decDis.backward(retain_graph=True)
                optimizer_decDis.step()
                
                #train recDis
                optimizer_decDis.zero_grad()
                optimizer_recDis.zero_grad()
                fake = torch.cat((fake_start, fake), 1) #add fake LR to WC
                fake = Rec(fake)
                real = labels_hr_dis[batch_size:, :, :, :]
                hr_test = torch.cat((fake, real), 0)
                 
                logit_recDis = recDis(hr_test)
                loss_recDis = BCE(logit_recDis, dis_labels)
                
                loss_recDis.backward()
                optimizer_recDis.step()

                if k == k_steps:
                    break

        if i == int(len(images)/batch_size)-1:
            break

 
    torch.save(Dec.state_dict(), RESU_PATH+'GAN_Dec'+str(epoch))
    torch.save(Rec.state_dict(), RESU_PATH+'GAN_Rec'+str(epoch))

    torch.save(decDis.state_dict(), RESU_PATH+'GAN_decDis'+str(epoch))
    torch.save(recDis.state_dict(), RESU_PATH+'GAN_recDis'+str(epoch))
