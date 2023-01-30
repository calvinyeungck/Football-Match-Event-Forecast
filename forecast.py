# -*- coding: utf-8 -*-
"""

ref https://github.com/statsonthecloud/Soccer-SEQ2Event/blob/main/Seq2Event_Notebook02_Modelling.ipynb 
paper https://eprints.soton.ac.uk/458099/
"""
 
#%%
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from datetime import datetime
#%%
'''
attention to transformer encoder


'''
train = pd.read_csv("train.csv")
valid = pd.read_csv("valid.csv")
test = pd.read_csv("test.csv")
test=pd.concat([train,valid,test], ignore_index=True)

#%% hyperparameter
window_size=40 #num of previous event considered
epochs=50 #num of epochs for model training
batch_size,num_workers=100 ,0 #num of samlpe per update, num of cpu

#%% features scaling

minmax= MinMaxScaler(feature_range = (0,1))
minmax_deltaT=MinMaxScaler(feature_range = (0,1))
train[['T','y', 'x', 's','deltay','deltax','sg','thetag','zone_s','zone_deltay','zone_deltax','zone_sg','zone_thetag']] = minmax.fit_transform(train[['T','y','x','s','deltay','deltax','sg','thetag','zone_s','zone_deltay','zone_deltax','zone_sg','zone_thetag']])
train[['deltaT']] = minmax_deltaT.fit_transform(train[['deltaT']])

valid[['T','y', 'x', 's','deltay','deltax','sg','thetag','zone_s','zone_deltay','zone_deltax','zone_sg','zone_thetag']] = minmax.fit_transform(valid[['T','y','x','s','deltay','deltax','sg','thetag','zone_s','zone_deltay','zone_deltax','zone_sg','zone_thetag']])
valid[['deltaT']] = minmax_deltaT.fit_transform(valid[['deltaT']])

test[['T','y', 'x', 's','deltay','deltax','sg','thetag','zone_s','zone_deltay','zone_deltax','zone_sg','zone_thetag']] = minmax.fit_transform(test[['T','y','x','s','deltay','deltax','sg','thetag','zone_s','zone_deltay','zone_deltax','zone_sg','zone_thetag']])
test[['deltaT']] = minmax_deltaT.fit_transform(test[['deltaT']])

#%% Define categorical action encode/decode
# set up idx2char and char2idx
action=['p', '_', 'd', 'x', 's']   # list of all unique characters in the text
num_chars_test = len(action)                   
char2idx_test = dict((c, i) for i, c in enumerate(action))
idx2char_test = dict((i, c) for i, c in enumerate(action))

num_chars=5 # hyperparameter
idx2char = {0: 'p', 1: '_', 2: 'd', 3: 'x', 4: 's'}
char2idx = {'p': 0, '_': 1, 'd': 2, 'x': 3, 's': 4}

# replace characters with numbers
train['act'].replace(char2idx,inplace=True)
valid['act'].replace(char2idx,inplace=True)
test['act'].replace(char2idx,inplace=True)
#%% zone label from 1 to 20, set to 0 to 19 
train.zone=train.zone-1
valid.zone=valid.zone-1
test.zone=test.zone-1


#%% Specify variables of interest

input_vars= ['act','deltaT','zone_s','zone_deltay','zone_deltax','zone_sg','zone_thetag','zone']
target_vars = ['deltaT', 'zone', 'act']

#%% Specify loss function weighting
zone=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
action=[0,1,2,3,4]
weight_action_class=compute_class_weight(
                                        class_weight = "balanced",
                                        classes = action,
                                        y = train.act                                                    
                                    )

weight_zone_class=compute_class_weight(
                                        class_weight = "balanced",
                                        classes = zone,
                                        y = train.zone                                                  
                                    )

weight_action_class = torch.tensor(weight_action_class)
weight_zone_class = torch.tensor(weight_zone_class)
weight_deltaT=torch.tensor([1.])
#%% Define valid slices

def valid_slice_flag(df,window_size):
    df["valid_slice_flag"]=True
    for i in range(len(df)-1):
        if df.MID[i] != df.MID[i+1]:
            df.loc[i+1-window_size:i+1,["valid_slice_flag"]]=False
    df.loc[len(df)-window_size:len(df),["valid_slice_flag"]]=False
    return  df
train=valid_slice_flag(train,window_size)
valid=valid_slice_flag(valid,window_size)
test=valid_slice_flag(test,window_size)

#%% data encodeing or reorder df and turn to tensor
other=['zone_s','zone_deltay','zone_deltax','zone_sg','zone_thetag']
def encode(df):
    #num_class_action=5
    #num_class_zone=20
    
    df_tensor_action=torch.from_numpy(df["act"].values)
    df_tensor_action=df_tensor_action.view(len(df_tensor_action),1)
    #one_hot_action = torch.nn.functional.one_hot(df_tensor_action, num_classes=num_class_action)
    
    df_tensor_zone=torch.from_numpy(df["zone"].values)
    df_tensor_zone=df_tensor_zone.view(len(df_tensor_zone),1)
    #one_hot_zone = torch.nn.functional.one_hot(df_tensor_zone, num_classes=num_class_zone)
    
    df_tensor_deltaT=torch.from_numpy(df['deltaT'].values)
    df_tensor_deltaT=df_tensor_deltaT.view(len(df_tensor_deltaT),1)
    
    df_tensor_other=torch.from_numpy(df[other].values)
    
    #encode_df=torch.cat((one_hot_action , one_hot_zone, df_tensor_deltaT,df_tensor_other),1)
    encode_df=torch.cat((df_tensor_action , df_tensor_zone, df_tensor_deltaT,df_tensor_other),1)
    
    #return encode_df,one_hot_action,one_hot_zone,df_tensor_deltaT,df_tensor_other
    return encode_df
#encode_train,encode_train_action,encode_train_zone,encode_train_deltaT,encode_train_other=encode(train)
encode_train=encode(train)
encode_valid=encode(valid)
encode_test=encode(test)
#%% define dataclass  

idx_all_train = np.repeat(True,len(train))
class train_data():
    def __init__(self,idx=idx_all_train):
        
        self.idx = idx
        self.valid_slice_idxn = np.where(np.logical_and(self.idx,train["valid_slice_flag"]))[0]  #in both the idx and has a valid slice flag
    def __len__(self):
        return int(np.sum(train.loc[self.idx,"valid_slice_flag"]))
    def __getitem__(self, i):
        j = self.valid_slice_idxn[i]
        x = encode_train[j:j+window_size]
        y = train.iloc[j+window_size].loc[target_vars]
        y = torch.from_numpy(y.to_numpy(dtype="float64"))
        return x,  y

idx_all_valid = np.repeat(True,len(valid)) 
class valid_data():
    def __init__(self,idx=idx_all_valid):
        
        self.idx = idx
        self.valid_slice_idxn = np.where(np.logical_and(self.idx,valid["valid_slice_flag"]))[0]  #in both the idx and has a valid slice flag
    def __len__(self):
        return int(np.sum(valid.loc[self.idx,"valid_slice_flag"]))
    def __getitem__(self, i):
        j = self.valid_slice_idxn[i]
        x = encode_valid[j:j+window_size]
        y = valid.iloc[j+window_size].loc[target_vars]
        y = torch.from_numpy(y.to_numpy(dtype="float64"))
        return x,  y
    
idx_all_test = np.repeat(True,len(test))   
class test_data():
    def __init__(self,idx=idx_all_test):
        
        self.idx = idx
        self.valid_slice_idxn = np.where(np.logical_and(self.idx,test["valid_slice_flag"]))[0]  #in both the idx and has a valid slice flag
    def __len__(self):
        return int(np.sum(test.loc[self.idx,"valid_slice_flag"]))
    def __getitem__(self, i):
        j = self.valid_slice_idxn[i]
        x = encode_test[j:j+window_size]
        y = test.iloc[j+window_size].loc[target_vars]
        y = torch.from_numpy(y.to_numpy(dtype="float64"))
        return x,  y

#%% model
def positional_encoding(src):
  # src = X_cat0; d_model = 15

  pos_encoding = torch.zeros_like(src)
  seq_len = pos_encoding.shape[0]
  d_model = pos_encoding.shape[1]

  for i in range(d_model):
    for pos in range(seq_len):
      if i % 2 == 0:
        pos_encoding[pos,i] = np.sin(pos/100**(2*i/d_model))
      else:
        pos_encoding[pos,i] = np.cos(pos/100**(2*i/d_model))
  # plt.imshow(pos_encoding.cpu().numpy())
  return pos_encoding.float()

#%%
class NMSTPP(nn.Module):
    def __init__(self):  #pick up all specification vars from the global environment
        super(NMSTPP, self).__init__()    
        # for action one-hot
        self.emb_act = nn.Embedding(action_emb_in,action_emb_out,scale_grad_by_freq=scale_grad_by_freq)
        # for zone one-hot
        self.emb_zone = nn.Embedding(zone_emb_in,zone_emb_out,scale_grad_by_freq=scale_grad_by_freq)
        # for continuous features
        self.lin0 = nn.Linear(other_lin_in,other_lin_out,bias=True) 

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_features_len,nhead=mutihead_attention,batch_first=True,dim_feedforward=hidden_dim).to(device)

        self.lin_relu = nn.Linear(input_features_len,input_features_len)
        self.lin_deltaT = nn.Linear(input_features_len,1)
        self.lin_zone = nn.Linear(input_features_len+1,20)
        self.lin_action = nn.Linear(input_features_len+1+20,5)
        self.NN_deltaT = nn.ModuleList()
        self.NN_zone = nn.ModuleList()
        self.NN_action = nn.ModuleList()
        for num_layer_deltaT in range(1):
            self.NN_deltaT.append(nn.Linear(input_features_len,input_features_len))
            #self.NN_deltaT.append(nn.ReLU())
            #self.NN_deltaT.append(nn.Dropout(p))
        for num_layer_zone in range(1):
            self.NN_zone.append(nn.Linear(input_features_len+1,input_features_len+1))
            #self.NN_zone.append(nn.ReLU())
            #self.NN_zone.append(nn.Dropout(p))
        for num_layer_action in range(2):
            self.NN_action.append(nn.Linear(input_features_len+1+20,input_features_len+1+20))
            #self.NN_action.append(nn.ReLU())
            #self.NN_action.append(nn.Dropout(p))


        print(self)        

    def forward(self, X):

        feed_action=X[:,:,0]
        feed_zone=X[:,:,1]
        feed_other_deltaT=X[:,:,2:]
        
        X_act = self.emb_act(feed_action.int())
        X_zone = self.emb_zone(feed_zone.int())
        feed_other_deltaT= self.lin0(feed_other_deltaT.float())
        X_cont= feed_other_deltaT
       
        X_cat = torch.cat((X_act,X_zone,X_cont),2)
        X_cat = X_cat.float()
     
        src = X_cat+ positional_encoding(X_cat).to(device)
     
        src=src.float()
        
        X_cat_seqnet= self.encoder_layer(src)
        x_relu=self.lin_relu(X_cat_seqnet[:,-1,:])

        model_deltaT=x_relu
        for layer in self.NN_deltaT[:]:
            model_deltaT=layer(model_deltaT)
        model_deltaT=self.lin_deltaT(model_deltaT)

        features_zone=torch.cat((model_deltaT, x_relu),1)
        model_zone=features_zone
        for layer in self.NN_zone[:]:
            model_zone=layer(model_zone)
        model_zone=self.lin_zone(model_zone)


        features_action=torch.cat((model_zone,model_deltaT, x_relu),1)
        model_action=features_action
        for layer in self.NN_action[:]:
            model_action=layer(model_action)
        model_action=self.lin_action(model_action)


        out=torch.cat((model_deltaT,model_zone,model_action),1)

        
        return out

#%% cost function

def cost_function(y,y_head,deltaT_weight,zone_weight,action_weight):
    #print("temp",y.size(),y_head.size())
    y_deltaT=y[:,0].float()
    y_zone=y[:,1].long()
    y_action=y[:,2].long()
    y_head_deltaT=y_head[:,0].float() 
    y_head_zone=y_head[:,1:21]
    y_head_action=y_head[:,21:]
    #print("action",y_action.size(),y_head_action.size())
    CEL_action = nn.CrossEntropyLoss(weight=weight_action_class.float().to(device),reduction ="none")
    Yhat_CEL_action  = torch.mean(CEL_action(y_head_action,y_action)) 
    #print("zone",y_zone.size(),y_head_zone.size())
    CEL_zone = nn.CrossEntropyLoss(weight=weight_zone_class.float().to(device),reduction="none")
    Yhat_CEL_zone  = torch.mean(CEL_zone( y_head_zone,y_zone)) 
    #print("deltaT",y_deltaT.size(),y_head_deltaT.size())
    Yhat_RMSE_deltaT= torch.mean((y_deltaT-y_head_deltaT)**2)**0.5
    
    Loss=  Yhat_RMSE_deltaT*deltaT_weight +Yhat_CEL_zone*zone_weight+Yhat_CEL_action*action_weight
    #print(Loss)
    #print(Yhat_MSE_deltaT)
    #print(Yhat_CEL_zone)
    #print(Yhat_CEL_action)
    return Loss,Yhat_RMSE_deltaT,Yhat_CEL_zone,Yhat_CEL_action

#%% dataloader 

train_dataset = train_data()
valid_dataset =valid_data()
test_dataset =test_data()
train_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size,num_workers=num_workers,drop_last=True)
valid_loader = DataLoader(valid_dataset,shuffle=False,batch_size=batch_size,num_workers=num_workers,drop_last=True)
test_loader = DataLoader(test_dataset,shuffle=False,batch_size=batch_size,num_workers=num_workers,drop_last=False)



#%% training,valid,test one epoch
def model_epoch(dataloader, model, optimiser, scheduler,epochtype):
    if epochtype=="train":
        model.train()     #turn training off if (val or test)
    else:
        model.eval()
     
    
    
    size = len(dataloader.dataset)
    loss_rollingmean, lossCEL_zone_rollingmean,lossCEL_action_rollingmean, lossRMSE_rollingmean  = 0., 0., 0., 0.
    for batch, (X, Y) in enumerate(dataloader):
        batch_total=len(train_loader)
        #print("batch",batch+1,"/", batch_total)
        X, Y = X.to(device), Y.to(device)
        
        pred = model(X)
        #print(Y)
        #print(pred)
        #print(pred.size())
        Loss,RMSE_deltaT,CEL_zone,CEL_action = cost_function(Y,pred,10,1,1)
        loss_rollingmean = loss_rollingmean+(Loss-loss_rollingmean)/(1+batch)
        lossCEL_zone_rollingmean =  lossCEL_zone_rollingmean+(CEL_zone-lossCEL_zone_rollingmean)/(1+batch)
        lossCEL_action_rollingmean = lossCEL_action_rollingmean+(CEL_action-lossCEL_action_rollingmean)/(1+batch)
        lossRMSE_rollingmean = lossRMSE_rollingmean+(RMSE_deltaT-lossRMSE_rollingmean)/(1+batch)
        
        if epochtype=="train":
            optimiser.zero_grad()
            Loss.backward()
            optimiser.step()
    
        if batch % 500 == 0:
            print("batch",batch,"/", batch_total)
            loss=Loss
            loss, current = loss.item(), batch * X.shape[0]
            print(f"loss: {loss:>7f}, ln(1+loss): {np.log(1+loss):>7f} | CEloss_zone: {CEL_zone:>7f},CEloss_action: {CEL_action:>7f}, MSEloss: {RMSE_deltaT} | batch: {batch} | sample: [{current:>5d}/{size:>5d}] | lr: {optimiser.param_groups[0]['lr']}")
    loss_rollingmean, lossCEL_zone_rollingmean, lossCEL_action_rollingmean,lossRMSE_rollingmean  = loss_rollingmean.detach().cpu().numpy().item(),  lossCEL_zone_rollingmean.detach().cpu().numpy().item(),lossCEL_action_rollingmean.detach().cpu().numpy().item(), lossRMSE_rollingmean.detach().cpu().numpy().item()  
    print("epoch ended")
    print(f"Epoch loss:    mean: {loss_rollingmean:>7f}, ln(1+loss) mean: {np.log(1+loss_rollingmean):>7f}")
    print(f"Epoch CEloss_zone:  mean: {lossCEL_zone_rollingmean:>7f}, ln(1+loss) mean: {np.log(1+lossCEL_zone_rollingmean):>7f}")
    print(f"Epoch CEloss_action:  mean: {lossCEL_action_rollingmean:>7f}, ln(1+loss) mean: {np.log(1+lossCEL_action_rollingmean):>7f}")
    print(f"Epoch MSEloss: mean: {lossRMSE_rollingmean:>7f}, ln(1+loss) mean: {np.log(1+lossRMSE_rollingmean):>7f}")
    return loss_rollingmean, lossCEL_zone_rollingmean,lossCEL_action_rollingmean, lossRMSE_rollingmean

#%% training,valid,test,mutiple epoch

def model_train(epochs):
  torch.cuda.empty_cache(); import gc; gc.collect()
  global model 
  model = NMSTPP().to(device)
  #optimiser  = optim.RMSprop(model.parameters(),lr=0.01,eps=1e-16)
  optimiser  = optim.Adam(model.parameters(),lr=0.01,eps=1e-16)
  scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,factor=.1,patience=3,verbose=True)

  #model.load_state_dict(torch.load("/content/gdrive/MyDrive/COMP6200project/Soccer/Data/1processed/MDLstate_20210730am_working_1GRU20210816star3"))

  trainloss_hist = pd.DataFrame(columns=["epoch","trn_L","trn_CEL_zone","trn_CEL_action","trn_MSEL"])
  time_start = datetime.now()
  for t in range(epochs):
      torch.cuda.empty_cache(); import gc; gc.collect()
      print(f"Epoch {t}\n-------------------------------")
      trainloss = model_epoch(train_loader, model, optimiser, scheduler,"train")   #TRAIN - the important bit!!
     

      epochloss = pd.DataFrame(np.concatenate((np.array([t]),np.asarray(trainloss)))).T
      epochloss.columns = trainloss_hist.columns
      trainloss_hist = pd.concat([trainloss_hist, epochloss], ignore_index=True)

      if optimiser.param_groups[0]["lr"] < 1E-7:
        break
  time_end = datetime.now()
  trainable_params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
  train_time=time_end-time_start
  train_time=train_time.total_seconds()
  with torch.no_grad():
    torch.cuda.empty_cache(); import gc; gc.collect()
    valid_loss=model_epoch(valid_loader, model, optimiser, scheduler,"valid")
  with torch.no_grad():
    torch.cuda.empty_cache(); import gc; gc.collect()
    test_loss=model_epoch(test_loader, model, optimiser, scheduler,"test")
  return trainloss_hist,train_time,trainable_params_num,model.state_dict(),valid_loss,test_loss

#%% function for predict

def predict(dataloader_x):
    with torch.no_grad():
        model.eval()                     
        i = 0
        for batch, (X, Y) in enumerate(dataloader_x):
            X = X.to(device)
            pred = model(X)
            if i == 0:
              all_pred = pred.detach().cpu()
            else:
              all_pred = torch.cat((all_pred,pred.detach().cpu()))
            i+=1
            if batch % 500 == 0:
                print("batch",batch,"/", len(dataloader_x))
    return all_pred




#%%
if __name__ == '__main__': 

    attention_layer=1 #numbers of attention layers
    hidden_dim=1024 #no. of hidden layer of the FFN
    mutihead_attention=1 #number of attention head
    scale_grad_by_freq=True #scale gradients by the inverse of frequency of the class


    action_emb_in=len(action) 
    action_emb_out=len(action) 
    zone_emb_in=len(zone)
    zone_emb_out=len(zone)
    other_lin_in=len(other)+1 #+1 for deltaT
    other_lin_out=len(other)+1 #+1 for deltaT
    input_features_len=action_emb_out+zone_emb_out+other_lin_out
    hist_dim=1+20+5 #output sample size of the NMSTPP model


    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = NMSTPP().to(device)
    model.load_state_dict(torch.load("model.pt"))

   
    test_pred_sub=predict(test_loader)

    test_pred_sub=pd.DataFrame(test_pred_sub.numpy())

    test_pred_sub["unminmax_deltaT"]=minmax_deltaT.inverse_transform(test_pred_sub[[0]])
    

    test_pred_sub.to_csv("all_pred.csv")
    
