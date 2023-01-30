# -*- coding: utf-8 -*-

#%%
import pandas as pd
import numpy as np
#%%
data = pd.read_csv("dataset.csv")
#%% create the hard label for Juego de Posición (position game)

data["zone"]=data[['zone_degree_1',
'zone_degree_2', 'zone_degree_3', 'zone_degree_4', 'zone_degree_5',
'zone_degree_6', 'zone_degree_7', 'zone_degree_8', 'zone_degree_9',
'zone_degree_10', 'zone_degree_11', 'zone_degree_12', 'zone_degree_13',
'zone_degree_14', 'zone_degree_15', 'zone_degree_16', 'zone_degree_17',
'zone_degree_18', 'zone_degree_19', 'zone_degree_20']].idxmax(axis=1).str.replace("zone_degree_", '')
data["zone"]=pd.to_numeric(data["zone"])

#%% create features
'''
 'zone_s', distance since previous event
 'zone_deltay', change in zone distance in x 
 'zone_deltax', change in zone distance in y
 'zone_sg',  distance to the center of opponent goal from the zone
 'zone_thetag' angle from the center of opponent goal 
'''
centroid_x=[ 8.5 , 25.25, 41.75, 58.25, 74.75, 91.5,8.5 , 25.25, 41.75, 58.25, 74.75, 
   91.5,33.5, 66.5,33.5, 66.5,33.5, 66.5,8.5,91.5]
centroid_y=[89.45, 89.45, 89.45, 89.45, 89.45, 89.45,10.55, 10.55, 10.55, 10.55, 10.55, 10.55,
   71.05, 71.05,50., 50.,28.95, 28.95, 50.,50.]

data["zone_s"]=""
data["zone_deltay"]=""
data["zone_deltax"]=""
data["zone_sg"]=""
data["zone_thetag"]=""


for i in range(len(data)):
    #print(i+1,"/",len(data))
    if i==0:
        data["zone_deltay"][i]=0
        data["zone_deltax"][i]=0
        data["zone_s"][i]=0
    else:
        data["zone_deltay"][i]=centroid_x[data["zone"][i]-1]-centroid_x[data["zone"][i-1]-1]
        data["zone_deltax"][i]=centroid_y[data["zone"][i]-1]-centroid_y[data["zone"][i-1]-1]
        data["zone_s"][i]=((data["zone_deltax"][i]*1.05)**2+(data["zone_deltay"][i]*0.68)**2)**0.5
    data["zone_sg"][i]=(((centroid_x[data["zone"][i]-1]-100)*1.05)**2+((centroid_y[data["zone"][i]-1]-50)*0.68)**2)**0.5
    data["zone_thetag"][i]=np.abs(np.arctan2((centroid_y[data["zone"][i]-1]-50)*0.68,(centroid_x[data["zone"][i]-1]-100)*1.05))

#df["s"] = ((df["deltax"]*1.05)**2+(df["deltay"]*0.68)**2)**0.5
#df["sg"]=(((df["x"]-100)*1.05)**2+((df["y"]-50)*0.68)**2)**0.5
#df["thetag"] = np.abs(np.arctan2((df["y"]-50)*0.68,(df["x"]-100)*1.05))


#%% reset s zone_s, delta x,y zone_delta x,y for the event at the start of each game to 0
for i in range(len(data)):
    #print(i+1,"/",len(data))
    if i == 0:
        data["s"][i]=0
        data["zone_s"][i]=0
        data["deltax"][i]=0
        data["deltay"][i]=0
        data["zone_deltax"][i]=0
        data["zone_deltay"][i]=0
    elif data["MID"][i] != data["MID"][i-1]:
        data["s"][i]=0
        data["zone_s"][i]=0
        data["deltax"][i]=0
        data["deltay"][i]=0
        data["zone_deltax"][i]=0
        data["zone_deltay"][i]=0
#%%
data.to_csv("featureset.csv",index=False)
#%% number of matches per league
#for i in np.unique(data[['comp']]):
#    temp=data[data['comp']==i]
#    print(i,temp.MID.nunique())
    
"""
Number of match
DE 300
EN 371
ES 371
FR 371
IT 371
Number of event
IT    531088
EN    522221
FR    511830
ES    507436
DE    420782
"""
#%% 
Train_ratio=0.8 #1428 matches
Valid_ratio=0.1 #178 matches
Test_ratio=0.1 #178 matches

Train_id=[]
Valid_id=[]
Test_id=[]
for i in np.unique(data[['comp']]):
    temp=data[data['comp']==i]
    id_list=temp.MID.unique()
    Train_id+=id_list[0:round(temp.MID.nunique()*Train_ratio)].tolist()
    Valid_id+=id_list[round(temp.MID.nunique()*Train_ratio):round(temp.MID.nunique()*(Train_ratio+Valid_ratio))].tolist()
    Test_id+=id_list[round(temp.MID.nunique()*(Train_ratio+Valid_ratio)):].tolist()
#    Train_id.append(id_list[0:round(temp.MID.nunique()*Train_ratio)])
#    Valid_id.append(id_list[round(temp.MID.nunique()*Train_ratio):round(temp.MID.nunique()*(Train_ratio+Valid_ratio))])
#    Test_id.append(id_list[round(temp.MID.nunique()*(Train_ratio+Valid_ratio)):])

#%% get the test valid and test dataset
train=data[data["MID"].isin(Train_id)]
valid=data[data["MID"].isin(Valid_id)]
test=data[data["MID"].isin(Test_id)]
#%%
train.to_csv("train.csv",index=False)
valid.to_csv("valid.csv",index=False)
test.to_csv("test.csv",index=False)














