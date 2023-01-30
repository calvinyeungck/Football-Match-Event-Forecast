# -*- coding: utf-8 -*-

import numpy as np; import pandas as pd; import sklearn as sk; import matplotlib.pyplot as plt; import seaborn as sns; import os
import random; from scipy import stats; from sklearn import preprocessing; from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import numpy.lib.recfunctions as rfn; import statsmodels.api as sm; from statsmodels.graphics.regressionplots import abline_plot
import numpy as np; from sklearn.linear_model import LinearRegression; from sklearn.metrics import mean_squared_error, r2_score
import math; from sklearn import linear_model
from numpy import linalg; from scipy import stats; import cv2; import math; from matplotlib import cm; from matplotlib import colors
import matplotlib as mpl; from matplotlib.ticker import (AutoMinorLocator, MultipleLocator); import scipy.stats as st; import scipy as sp
mpl.rcParams['figure.dpi'] = 150 #default is 72.0
import os; print(os.getcwd())
from os import listdir
from os.path import isfile, join
from PIL import Image
from matplotlib import cm
import matplotlib.patches as mpatches
from tqdm import tqdm
from zipfile import ZipFile
import matplotlib.pyplot as plt
import mplsoccer
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import VerticalPitch, FontManager, Sbopen

#%% get data with the following code
#!wget "https://figshare.com/ndownloader/files/14464685/events.zip"
#ZipFile("events.zip","r").extractall("./")
#%% load JSON file get the event from England,France,Germany, Italy,Spain league (excludeing workd cup and wuropean championship)
json_files = ["events_England.json",
              "events_France.json",
              "events_Germany.json",
              "events_Italy.json",
              "events_Spain.json"]

json_codes = ["EN","FR","DE","IT","ES"]

for i in range(len(json_files[:])):
  print(i+1)    
  dfraw = pd.read_json(json_files[i])
  dfred = dfraw[["teamId","matchId","eventSec","matchPeriod","positions","eventName","subEventName","tags"]]
  dfred.insert(0,"comp",json_codes[i])
  if i == 0: df = pd.DataFrame.copy(dfred)
  if i != 0: df = pd.concat([df,dfred])

df = df.reset_index(drop=True)
dforig = df.copy()      # all operations performed on df, but keeping an original just in case we need to refer to it later

#%% Basic trimming

# identify goals and own-goals from the tags metadata
#####
df["goal"] = ""

idx = df["tags"].astype(str).str.contains("\{'id': 101\}")
df.loc[idx,"goal"] = "goal"

idx = df["tags"].astype(str).str.contains("\{'id': 102\}")
df.loc[idx,"goal"] = "own-goal"

# identify offensive dribbles from the tags metadata
#####
df["off_dribble"] = False

idx0 = df["subEventName"] == "Ground attacking duel"
idx1 = df["tags"].astype(str).str.contains("\{'id': 501\}")
idx2 = df["tags"].astype(str).str.contains("\{'id': 502\}")
idx3 = df["tags"].astype(str).str.contains("\{'id': 503\}")
idx4 = df["tags"].astype(str).str.contains("\{'id': 504\}")
idx5 = np.logical_or.reduce((idx1, idx2, idx3, idx4))
idx6 = np.logical_and(idx0,idx5)
df.loc[idx6,"off_dribble"] = True


# identify successful touches
#####
df["good_tch"] = False

idx0 = df["subEventName"] == "Touch"
idx1 = df["tags"].astype(str).str.contains("\[\]")
idx2 = np.logical_and(idx0,idx1)
df.loc[idx2,"good_tch"] = True


df = df.drop("tags",axis=1)

# process co-ordinate data
#####
df["positions"] = df["positions"].astype(str).str.replace(r"\[\{'y': ","")
df["positions"] = df["positions"].astype(str).str.replace(r" 'x': ","")
df["positions"] = df["positions"].astype(str).str.replace(r"\}, \{'y': ",",")
df["positions"] = df["positions"].astype(str).str.replace(r"\}\]","")
dfpos = df["positions"].str.split(',', expand=True)
dfpos.columns = ["y0","x0","y1","x1"]
dfpos = dfpos.loc[:,["x0","y0","x1","y1"]]

df = pd.concat([df, dfpos], axis=1)
df = df.drop("positions",axis=1)

#%% Start second half time at 60 minutes

#np.unique(df['matchPeriod'])
# np.max(df.loc[df['matchPeriod']=="1H","eventSec"])   # 3302s = 55m
# np.max(df.loc[df['matchPeriod']=="2H","eventSec"])   # 3537s = 59m
# np.max(df.loc[df['matchPeriod']=="E1","eventSec"])   # 1117s = 18m
# np.max(df.loc[df['matchPeriod']=="E2","eventSec"])   # 1163s = 18m
# np.max(df.loc[df['matchPeriod']=="P","eventSec"])    # 680s  = 12m

df.loc[df['matchPeriod']=="2H","eventSec"] = df.loc[df['matchPeriod']=="2H","eventSec"] + 60*60    #start second half time at 60 minutes
df.loc[df['matchPeriod']=="E1","eventSec"] = df.loc[df['matchPeriod']=="E1","eventSec"] + 120*60   #start ET1 at 120 minutes
df.loc[df['matchPeriod']=="E2","eventSec"] = df.loc[df['matchPeriod']=="E2","eventSec"] + 150*60   #start ET2 at 150 minutes
df.loc[df['matchPeriod']=="P","eventSec"]  = df.loc[df['matchPeriod']=="P","eventSec"] + 180*60   #start Pens at 180 minutes
df = df.drop("matchPeriod",axis=1)

#%% Make goal and own-goal events

idx = np.where(df["goal"] == "goal")[0]

for i in range(len(idx)):
    temprow = pd.DataFrame.copy(df.iloc[idx[i],:])
    temprow["subEventName"] = "goal"
    temprow = pd.DataFrame(temprow).T

    if i == 0: rowstoadd = temprow
    if i != 0: rowstoadd = pd.concat([rowstoadd,temprow])

idx = np.where(df["goal"] == "own-goal")[0]

for i in range(len(idx)):
    temprow = pd.DataFrame.copy(df.iloc[idx[i],:])
    temprow["subEventName"] = "own-goal"
    temprow = pd.DataFrame(temprow).T

    rowstoadd = pd.concat([rowstoadd,temprow])

df = pd.concat([df,rowstoadd])

#%% Sort df (so things are back in event order after the goal concatenation above)


df = df.sort_values(by=["comp","matchId","eventSec"])

#%% Filter out unwanted actions and map WyScout actions according to my project
#     'initial encoding' scheme. This is a simplification of the WyScout actions,
#     however later, the actions are simplified further.


df["actcomb"] = df["eventName"] + "_" + df["subEventName"]

#np.unique(df["act"])

simple_actions = ['Foul_Foul','Foul_Hand foul','Foul_Late card foul','Foul_Out of game foul','Foul_Protest','Foul_Simulation','Foul_Time lost foul',
                  'Foul_Violent Foul','Offside_','Free Kick_Corner','Free Kick_Free Kick','Free Kick_Free kick cross','Free Kick_Free kick shot',
                  'Free Kick_Goal kick','Free Kick_Penalty','Free Kick_Throw in','Pass_Cross','Pass_Hand pass','Pass_Head pass','Pass_High pass',
                  'Pass_Launch','Pass_Simple pass','Pass_Smart pass','Shot_Shot',
                  'Shot_goal','Free Kick_goal','Others on the ball_own-goal','Pass_own-goal',
                  'Duel_Ground attacking duel',
                  'Others on the ball_Acceleration','Others on the ball_Clearance','Others on the ball_Touch']
idx = np.isin(df["actcomb"],simple_actions)
df = df.loc[idx,:]

df["act"] = ""
##### x
idx = np.isin(df["actcomb"],['Foul_Foul','Foul_Hand foul','Foul_Late card foul','Foul_Out of game foul',
                         'Foul_Protest','Foul_Simulation','Foul_Time lost foul','Foul_Violent Foul','Offside_'])
df.loc[idx,"act"] = "x"

##### 0
idx = df["actcomb"]=='Free Kick_Goal kick'
df.loc[idx,"act"] = "0"

##### 1
idx = df["actcomb"]=='Free Kick_Throw in'
df.loc[idx,"act"] = "1"

##### 2
idx = df["actcomb"]=='Free Kick_Corner'
df.loc[idx,"act"] = "2"

##### 3
idx = df["actcomb"]=='Free Kick_Free Kick'
df.loc[idx,"act"] = "3"

##### 4
idx = df["actcomb"]=='Free Kick_Free kick cross'
df.loc[idx,"act"] = "4"

##### 5
idx = df["actcomb"]=='Free Kick_Free kick shot'
df.loc[idx,"act"] = "5"

##### 6
idx = df["actcomb"]=='Free Kick_Penalty'
df.loc[idx,"act"] = "6"

##### c
idx = df["actcomb"]=='Pass_Cross'
df.loc[idx,"act"] = "c"

##### p
idx = np.isin(df["actcomb"],['Pass_Hand pass','Pass_Head pass','Pass_High pass','Pass_Launch','Pass_Simple pass','Pass_Smart pass'])
df.loc[idx,"act"] = "p"

##### s
idx = df["actcomb"]=='Shot_Shot'
df.loc[idx,"act"] = "s"

##### g
idx = np.isin(df["actcomb"],['Shot_goal','Free Kick_goal'])
df.loc[idx,"act"] = "g"

##### h
idx = np.isin(df["actcomb"],['Others on the ball_own-goal','Pass_own-goal'])
df.loc[idx,"act"] = "h"

##### d
idx = df['off_dribble'] == True
df.loc[idx,"act"] = "d"

##### t
idx = np.isin(df["actcomb"],['Others on the ball_Acceleration'])
df.loc[idx,"act"] = "t"

idx = df['good_tch'] == True
df.loc[idx,"act"] = "t"

##### o
idx = np.isin(df["actcomb"],['Others on the ball_Clearance'])
df.loc[idx,"act"] = "o"


#%% Determine possession (and filter out actions not by the team in possession)


df["pos"] = 0
df = df.reset_index(drop=True)

for i in range(df.shape[0]):
  if i == 0:
    df["pos"][0] = df["teamId"][0]
  else:
    prevpos = df["pos"][i-1]
    curact  = df["act"][i]
    curteam = df["teamId"][i]
    if prevpos == curteam:       #if the team that previously was classified as being in possession now has possession then let's classify that possession as continuing
      df["pos"][i] = curteam
    else:
      if np.isin(curact,['0','1','2','3','4','5','6','c','p','s','g','t','d','o']):
        df["pos"][i] = curteam
      else:
        df["pos"][i] = prevpos
#most possession ends up just being the same as the team who is performing the action (in our truncated list of actions)
#but there are occassions where possession is retained despite an action from the other team - in particular when the other team fouls

#filter out actions not by the team in possession
df = df.loc[df["teamId"] == df["pos"],:]
df = df.reset_index(drop=True)
#remove events that aren't part of the scheme
idx = df['act'] != ""
df = df.loc[idx,:]
df = df.reset_index(drop=True)

#%% Tidying up to improve performance


df = df.drop(["eventName","subEventName"],axis=1)
df["eventSec"] = df["eventSec"].astype(int)
df = df.loc[:,["comp","matchId","teamId","eventSec","pos","act","x0","y0","x1","y1"]]

#%% Determine score difference

df3 = df.copy()   #backup

##########
#make a matchlog - which team played which during each match
df["matchteam"] = df["matchId"].astype(str) + "_" + df["teamId"].astype(str)
matchteams = np.unique(df["matchteam"])
matchteams = np.array(matchteams,dtype=str)

matchlog = pd.DataFrame(matchteams,columns=["matchteam"])
matchlog.insert(1,"team_vs",0)

for i in range(len(matchlog)):
  #i = 0
  temp0 = matchlog.loc[i,"matchteam"]
  temp0match = np.str.split(temp0,"_")[0]
  temp1matchteams = matchteams[np.flatnonzero(np.core.defchararray.find(matchteams,temp0match+"_")!=-1)]
  matchlog.loc[i,"team_vs"] = str(temp1matchteams[temp1matchteams != temp0][0])
  if i % 100 == 0: print(i)

matchlog['team_vs2'] = matchlog['team_vs'].str.split('_').str[1]

##########
#assign team played against
df = pd.merge(df,matchlog.loc[:,["matchteam","team_vs2"]],left_on="matchteam",right_on="matchteam",how="left")
df = df.rename({'team_vs2':'team_vs'},axis=1)

df.team_vs = pd.to_numeric(df.team_vs)

##########
#assign score advantage
df["scrad"] = 0
matches = np.unique(df["matchId"])
for j in range(len(matches)):
  #j=0
  m = matches[j]
  print(m,j)

  matchidx = df["matchId"] == m
  matchidxn = np.where(matchidx)[0]
  match_start_idxn = min(matchidxn)
  match_end_idxn   = max(matchidxn)

  #assign scrad for goals
  goals_idxn = np.where(np.logical_and(df["act"]=="g",df["matchId"] == m))[0]
  num_goals = len(goals_idxn)
  for i in range(num_goals):
    goal_idxn = goals_idxn[i]
    goal_scored_by = df.loc[goal_idxn]["teamId"]
    df.loc[np.logical_and(np.in1d(np.arange(len(df)), np.arange(goal_idxn,match_end_idxn+1)),
                          df["teamId"]==goal_scored_by),
          "scrad"] += 1
    df.loc[np.logical_and(np.in1d(np.arange(len(df)), np.arange(goal_idxn,match_end_idxn+1)),
                        df["team_vs"]==goal_scored_by),
          "scrad"] -= 1

  #assign scrad for own-goals
  goals_idxn = np.where(np.logical_and(df["act"]=="h",df["matchId"] == m))[0]   #same code as above, just now act type "h" (own-goal), as opposed to "g" (goal)
  num_goals = len(goals_idxn)
  for i in range(num_goals):
    print(i)
    goal_idxn = goals_idxn[i]
    goal_scored_by = df.loc[goal_idxn]["team_vs"]                               #team_vs get the goal, when it is an own-goal
    df.loc[np.logical_and(np.in1d(np.arange(len(df)), np.arange(goal_idxn,match_end_idxn+1)),
                          df["teamId"]==goal_scored_by),
          "scrad"] += 1
    df.loc[np.logical_and(np.in1d(np.arange(len(df)), np.arange(goal_idxn,match_end_idxn+1)),
                        df["team_vs"]==goal_scored_by),
          "scrad"] -= 1
  
  if j % 100 == 0: print(j)
#%% Distance and bearing

df.x0 = pd.to_numeric(df.x0)
df.y0 = pd.to_numeric(df.y0)
df.x1 = pd.to_numeric(df.x1)
df.y1 = pd.to_numeric(df.y1)

x00 = df[1:]["x0"].to_numpy()*1.05     #to scale for the actual dimensions of a soccer pitch...
x01 = df[:-1]["x0"].to_numpy()*1.05
y00 = df[1:]["y0"].to_numpy()*0.68
y01 = df[:-1]["y0"].to_numpy()*0.68

s = ((x00-x01)**2+(y00-y01)**2)**0.5    
s = np.append(s,0)
df["dist"] = s

theta = np.arctan2(y00-y01,x00-x01)/np.pi*180
theta = np.append(theta,0)
df["theta"] = theta


matches = np.unique(df["matchId"])
j=1
for j in range(len(matches)):
  m = matches[j]
  matchidx = df["matchId"] == m
  matchidxn = np.where(matchidx)[0]
  #match_start_idxn = min(matchidxn)
  match_end_idxn   = max(matchidxn)
  df.loc[match_end_idxn,"dist"]=0
  df.loc[match_end_idxn,"theta"]=0

  if j % 100 == 0: print(j)

################################################################################################
# Time difference
################################################################################################
t00 = df[:-1]["eventSec"].to_numpy()
t01 = df[1:]["eventSec"].to_numpy()


t = t01-t00
t = np.append(0,t)
t[t>60] = 60
t[t<0] = 0

df["deltaT"] = t

#%% Add phase tags

p00 = df[1:]["pos"].to_numpy()
p01 = df[:-1]["pos"].to_numpy()

new_possession_idxn = np.where(p00 != p01)[0]+1   

df["posID"] = 0

posID = np.zeros(df.shape[0])
i=0
for p in new_possession_idxn:
  posID[p:]+=1
  i += 1
  if i % 1000 == 0: print(i)

df["posID"] = posID
df["posID"] = df["posID"].astype(int)


##########################################################################################
#  PRE-PROCESS THE DATA
##########################################################################################
#trim, re-arrange
df = df.drop(["teamId","x1","y1","matchteam","team_vs"],axis=1)
df = df.loc[:,["comp","matchId","pos","posID","act","eventSec","deltaT","x0","y0","dist","theta","scrad"]]
df.columns = ["comp","MID","TID","PID","act","T","deltaT","x0","y0","s","theta","scrad"]


# add a row in after possession change
temp0 = df["PID"].to_numpy()
chgpos_idxn = np.where(temp0[1:] != temp0[:-1])[0]+1

idxn_small = np.array([])
idxn = np.array([])
for i in range(len(chgpos_idxn)):
  if i == 0:
    slice_start = 0
  else:
    slice_start = slice_end+1
  slice_end = chgpos_idxn[i]-1
  idxn_small = np.append(idxn_small,
                   np.append(np.arange(slice_start,slice_end+1),slice_end))
  
  if np.logical_or(i % 5000 == 0,i == len(chgpos_idxn)-1):
    print(i)
    print("appending idxn_small to idxn, and clearing idxn_small")
    idxn = np.append(idxn,idxn_small)
    idxn_small = np.array([])

df = df.iloc[idxn]
df = df.reset_index(drop=True)

# change action to '_' as last action of possession, but keep everything else the same
temp02 = df["PID"].to_numpy()
chgpos_idxn2 = np.where(temp02[1:] != temp02[:-1])[0]
df.loc[chgpos_idxn2,"act"] = "_"
df.loc[chgpos_idxn2,"deltaT"] = 0
df.loc[chgpos_idxn2,"s"] = 0
df.loc[chgpos_idxn2,"theta"] = 0.5


#%% FEATURE ENGINEERING: TIDY + SCALE

# Column tidy-up: re-arrange column order, and use slightly shorter names
##########
# df = df.loc[:,["comp","TID","MID","PID","act","T","deltaT","x0","y0","s","theta","scrad"]]
df = df.loc[:,["comp","TID","MID","PID","act","T","deltaT","y0","x0","s","theta","scrad"]]   # replace with the above - new nbook01
df.columns = ['comp', 'TID', 'MID', 'PID', 'act', 'T', 'deltaT', 'x', 'y', 's', 'theta', 'scrad']


#simplify actions further: apply simplified actions schema
##########
df = df.loc[df["act"]!="x",:]
df = df.reset_index(drop=True)
df.loc[df["act"]=="0","act"] = "p"
df.loc[df["act"]=="1","act"] = "p"
df.loc[df["act"]=="2","act"] = "x"    
df.loc[df["act"]=="3","act"] = "p"
df.loc[df["act"]=="4","act"] = "x"
df.loc[df["act"]=="5","act"] = "s"
df.loc[df["act"]=="6","act"] = "s"
df.loc[df["act"]=="t","act"] = "d"
df.loc[df["act"]=="o","act"] = "p"
df.loc[df["act"]=="c","act"] = "x"

# add deltax, deltay
##########
before = df["x"][:-1].to_numpy()
now    = df["x"][1:].to_numpy()
delta  = now-before
delta  = np.append(np.array(0.),delta)
df["deltax"] = delta

before = df["y"][:-1].to_numpy()
now    = df["y"][1:].to_numpy()
delta  = now-before
delta  = np.append(np.array(0.),delta)
df["deltay"] = delta

# recompute deltat: version above was scaled, but leaving scaling same as T aids comprehension when trying performing diagnostics
##########
before = df["T"][:-1].to_numpy()
now    = df["T"][1:].to_numpy()
delta  = now-before
delta  = np.append(np.array(0.),delta)
df["deltaT"] = delta
#df.loc[df["deltaT"] < 0,"deltaT"] = 0.
#df.loc[df["deltaT"] > 0.01,"deltaT"] = 0.01

# drop theta
##########
df = df.drop("theta",axis=1)   #we don't need theta any more - deltax0 and deltay0 have this feature covered, and are better than theta

# recompute s: version above was scaled, but leaving raw aids comprehension when trying performing diagnostics
##########
df["s"] = ((df["deltax"]*1.05)**2+(df["deltay"]*0.68)**2)**0.5
##########
#opponent's goal is at x=1.0, y=0.5
# add sg, thetag: distance and angle to goal
df["sg"]=(((df["x"]-100)*1.05)**2+((df["y"]-50)*0.68)**2)**0.5
df["thetag"] = np.abs(np.arctan2((df["y"]-50)*0.68,(df["x"]-100)*1.05))

#%% swap back x and y to wyscout format and recalculate some of the features
df.columns = ['comp', 'TID', 'MID', 'PID', 'act', 'T', 'deltaT', 'y', 'x', 's',
       'scrad', 'deltay', 'deltax', 'sg', 'thetag']
df["s"] = ((df["deltax"]*1.05)**2+(df["deltay"]*0.68)**2)**0.5
df["sg"]=(((df["x"]-100)*1.05)**2+((df["y"]-50)*0.68)**2)**0.5
df["thetag"] = np.abs(np.arctan2((df["y"]-50)*0.68,(df["x"]-100)*1.05))

#%% remove game that consist of owngoal
game_w_h=df.loc[df['act'] == "h", 'MID']
df2=df[:]
for i in game_w_h:
    df2=df2.drop(df2[df2.MID==i].index)

#%%get the row before goal and create features goal
df2["goal"]=0
temp=df2[df2.act=="g"].index
temp=temp-1
df2.loc[temp, "goal"] = 1

#%% remove row with action g 
df2=df2.drop(df2[df2.act=="g"].index)
#%% fix deltaT error (<0) due to new match and limit the maximum time to 60
df2.loc[df2['deltaT'] <0,'deltaT']=0 
df2.loc[df2['deltaT'] >60,'deltaT']=60 

#%% create c mean clustering for the coordinate, m=2 how fuzzy
df3=df2[:]
centroid_x=[ 8.5 , 25.25, 41.75, 58.25, 74.75, 91.5,8.5 , 25.25, 41.75, 58.25, 74.75, 
   91.5,33.5, 66.5,33.5, 66.5,33.5, 66.5,8.5,91.5]
centroid_y=[89.45, 89.45, 89.45, 89.45, 89.45, 89.45,10.55, 10.55, 10.55, 10.55, 10.55, 10.55,
   71.05, 71.05,50., 50.,28.95, 28.95, 50.,50.]

for i in range(20):
    dist=((df3["x"]-centroid_x[i])**2+(df3["y"]-centroid_y[i])**2)**0.5
    name="zone_dist_"+str(i+1)
    df3[name]=dist
    #print(dist)

for i in range(20):
    degree=0
    for j in range(20):
        degree+=(df3["zone_dist_"+str(i+1)]/df3["zone_dist_"+str(j+1)])**2
    degree=1/degree
    name="zone_degree_"+str(i+1)
    df3[name]=degree
    

#%% Output the dataset
df3.to_csv("dataset.csv",index=False)

#%% Plots
df = df3[:]
 
#%% distribution plot 
temp=df.PID.value_counts() #number of action in each possion
temp=temp.to_frame()
temp=temp.PID.value_counts()
temp=temp.to_frame()
temp.reset_index(inplace=True)
 
plt.bar(temp["index"],temp["PID"])
plt.plot(temp["index"],temp["PID"])
plt.ylabel("Frequency")
plt.xlabel("Number of action in an possession")
plt.title = ('Distribution of Number of Actions per Possession')
plt.margins(x=0.01, y=0.1)
#%%

df_p=df.loc[df['act'] == "p"]
df_end=df.loc[df['act'] == "_"]
df_d=df.loc[df['act'] == "d"]
df_x=df.loc[df['act'] == "x"]
df_s=df.loc[df['act'] == "s"]
#%%
# heatmap for p
pitch=mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")
fig,ax=pitch.draw(figsize=(8,16))
fig.set_facecolor("black")

pitch.kdeplot(df_p["x"],(100-df_p["y"]),ax=ax,cmap="Reds",shade=True)

# heatmap for _
pitch=mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")
fig,ax=pitch.draw(figsize=(8,16))
fig.set_facecolor("black")

pitch.kdeplot(df_end["x"],(100-df_end["y"]),ax=ax,cmap="Reds",shade=True)
# heatmap for d
pitch=mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")
fig,ax=pitch.draw(figsize=(8,16))
fig.set_facecolor("black")

pitch.kdeplot(df_d["x"],(100-df_d["y"]),ax=ax,cmap="Reds",shade=True)
# heatmap for x
pitch=mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")
fig,ax=pitch.draw(figsize=(8,16))
fig.set_facecolor("black")

pitch.kdeplot(df_x["x"],(100-df_x["y"]),ax=ax,cmap="Reds",shade=True)
# heatmap for s
pitch=mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")
fig,ax=pitch.draw(figsize=(8,16))
fig.set_facecolor("black")

pitch.kdeplot(df_s["x"],(100-df_s["y"]),ax=ax,cmap="Reds",shade=True)
#%% Heatmap Juego de Posición (position game) with scatter
# draw s
path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()]
pitch = mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")


fig,ax=pitch.draw(figsize=(8,16))
bin_statistic = pitch.bin_statistic_positional(df_s["x"],(100-df_s["y"]), statistic='count',
                                               positional='full', normalize=True)
pitch.heatmap_positional(bin_statistic, ax=ax, cmap='coolwarm', edgecolors='#22312b')
pitch.scatter(df_s["x"],(100-df_s["y"]), c='white', s=2, ax=ax)
labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18, ax=ax, ha='center', va='center',str_format='{:.0%}', path_effects=path_eff)

fig.set_facecolor("black")

# draw x
path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()]
pitch = mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")

fig,ax=pitch.draw(figsize=(8,16))
bin_statistic = pitch.bin_statistic_positional(df_x["x"],(100-df_x["y"]), statistic='count',
                                               positional='full', normalize=True)
pitch.heatmap_positional(bin_statistic, ax=ax, cmap='coolwarm', edgecolors='#22312b')
pitch.scatter(df_x["x"],(100-df_x["y"]), c='white', s=2, ax=ax)
labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18, ax=ax, ha='center', va='center',str_format='{:.0%}', path_effects=path_eff)

fig.set_facecolor("black")


# draw d
path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()]
pitch = mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")
fig,ax=pitch.draw(figsize=(8,16))
bin_statistic = pitch.bin_statistic_positional(df_d["x"],(100-df_d["y"]), statistic='count',
                                               positional='full', normalize=True)
pitch.heatmap_positional(bin_statistic, ax=ax, cmap='coolwarm', edgecolors='#22312b')
pitch.scatter(df_d["x"],(100-df_d["y"]), c='white', s=2, ax=ax)
labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18, ax=ax, ha='center', va='center',str_format='{:.0%}', path_effects=path_eff)

fig.set_facecolor("black")


# draw _
path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()]
pitch = mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")
fig,ax=pitch.draw(figsize=(8,16))
bin_statistic = pitch.bin_statistic_positional(df_end["x"],(100-df_end["y"]), statistic='count',
                                               positional='full', normalize=True)
pitch.heatmap_positional(bin_statistic, ax=ax, cmap='coolwarm', edgecolors='#22312b')
pitch.scatter(df_end["x"],(100-df_end["y"]), c='white', s=2, ax=ax)
labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18, ax=ax, ha='center', va='center',str_format='{:.0%}', path_effects=path_eff)

fig.set_facecolor("black")
# draw pass
path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()]
pitch = mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")
fig,ax=pitch.draw(figsize=(8,16))
bin_statistic = pitch.bin_statistic_positional(df_p["x"],(100-df_p["y"]), statistic='count',
                                               positional='full', normalize=True)
pitch.heatmap_positional(bin_statistic, ax=ax, cmap='coolwarm', edgecolors='#22312b')
pitch.scatter(df_p["x"],(100-df_p["y"]), c='white', s=2, ax=ax)
labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18, ax=ax, ha='center', va='center',str_format='{:.0%}', path_effects=path_eff)

fig.set_facecolor("black")
#%% Heatmap Juego de Posición (position game)
# draw s
path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()]
pitch = mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")


fig,ax=pitch.draw(figsize=(8,16))
bin_statistic = pitch.bin_statistic_positional(df_s["x"],(100-df_s["y"]), statistic='count',
                                               positional='full', normalize=True)
pitch.heatmap_positional(bin_statistic, ax=ax, cmap='coolwarm', edgecolors='#22312b')
labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18, ax=ax, ha='center', va='center',str_format='{:.0%}', path_effects=path_eff)

fig.set_facecolor("black")

# draw x
path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()]
pitch = mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")

fig,ax=pitch.draw(figsize=(8,16))
bin_statistic = pitch.bin_statistic_positional(df_x["x"],(100-df_x["y"]), statistic='count',
                                               positional='full', normalize=True)
pitch.heatmap_positional(bin_statistic, ax=ax, cmap='coolwarm', edgecolors='#22312b')
labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18, ax=ax, ha='center', va='center',str_format='{:.0%}', path_effects=path_eff)

fig.set_facecolor("black")


# draw d
path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()]
pitch = mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")
fig,ax=pitch.draw(figsize=(8,16))
bin_statistic = pitch.bin_statistic_positional(df_d["x"],(100-df_d["y"]), statistic='count',
                                               positional='full', normalize=True)
pitch.heatmap_positional(bin_statistic, ax=ax, cmap='coolwarm', edgecolors='#22312b')
labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18, ax=ax, ha='center', va='center',str_format='{:.0%}', path_effects=path_eff)

fig.set_facecolor("black")

# draw _
path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()]
pitch = mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")
fig,ax=pitch.draw(figsize=(8,16))
bin_statistic = pitch.bin_statistic_positional(df_end["x"],(100-df_end["y"]), statistic='count',
                                               positional='full', normalize=True)
pitch.heatmap_positional(bin_statistic, ax=ax, cmap='coolwarm', edgecolors='#22312b')
labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18, ax=ax, ha='center', va='center',str_format='{:.0%}', path_effects=path_eff)

fig.set_facecolor("black")
# draw pass
path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()]
pitch = mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")
fig,ax=pitch.draw(figsize=(8,16))
bin_statistic = pitch.bin_statistic_positional(df_p["x"],(100-df_p["y"]), statistic='count',
                                               positional='full', normalize=True)
pitch.heatmap_positional(bin_statistic, ax=ax, cmap='coolwarm', edgecolors='#22312b')
labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18, ax=ax, ha='center', va='center',str_format='{:.0%}', path_effects=path_eff)

fig.set_facecolor("black")



#%% Plot Juego de Posición (position game) grid cell only
grid_cell=[{'statistic': ([[0., 0., 0., 0., 0., 0.]]),
  'x_grid': ([[  0. ,  17. ,  33.5,  50. ,  66.5,  83. , 100. ],
         [  0. ,  17. ,  33.5,  50. ,  66.5,  83. , 100. ]]),
  'y_grid': ([[100. , 100. , 100. , 100. , 100. , 100. , 100. ],
         [ 78.9,  78.9,  78.9,  78.9,  78.9,  78.9,  78.9]]),
  'cx': ([ 8.5 , 25.25, 41.75, 58.25, 74.75, 91.5 ]),
  'cy': ([89.45, 89.45, 89.45, 89.45, 89.45, 89.45]),
  'binnumber': None,
  'inside': None},
 {'statistic': ([[0., 0., 0., 0., 0., 0.]]),
  'x_grid': ([[  0. ,  17. ,  33.5,  50. ,  66.5,  83. , 100. ],
         [  0. ,  17. ,  33.5,  50. ,  66.5,  83. , 100. ]]),
  'y_grid': ([[21.1, 21.1, 21.1, 21.1, 21.1, 21.1, 21.1],
         [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]]),
  'cx': ([ 8.5 , 25.25, 41.75, 58.25, 74.75, 91.5 ]),
  'cy': ([10.55, 10.55, 10.55, 10.55, 10.55, 10.55]),
  'binnumber': None,
  'inside': None},
 {'statistic': ([[0., 0.],
         [0., 0.],
         [0., 0.]]),
  'x_grid': ([[17., 50., 83.],
         [17., 50., 83.],
         [17., 50., 83.],
         [17., 50., 83.]]),
  'y_grid': ([[78.9, 78.9, 78.9],
         [63.2, 63.2, 63.2],
         [36.8, 36.8, 36.8],
         [21.1, 21.1, 21.1]]),
  'cx': ([[33.5, 66.5],
         [33.5, 66.5],
         [33.5, 66.5]]),
  'cy': ([[71.05, 71.05],
         [50.  , 50.  ],
         [28.95, 28.95]]),
  'binnumber': None,
  'inside': None},
 {'statistic': ([[0.]]),
  'x_grid': ([[ 0., 17.],
         [ 0., 17.]]),
  'y_grid': ([[78.9, 78.9],
         [21.1, 21.1]]),
  'cx': ([[8.5]]),
  'cy': ([[50.]]),
  'binnumber': None,
  'inside': None},
 {'statistic': ([[0.]]),
  'x_grid': ([[ 83., 100.],
         [ 83., 100.]]),
  'y_grid': ([[78.9, 78.9],
         [21.1, 21.1]]),
  'cx': ([[91.5]]),
  'cy': ([[50.]]),
  'binnumber': None,
  'inside': None}]

path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()]
pitch = mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")

# draw
x=[ 8.5 , 25.25, 41.75, 58.25, 74.75, 91.5,8.5 , 25.25, 41.75, 58.25, 74.75, 
   91.5,33.5, 66.5,33.5, 66.5,33.5, 66.5,8.5,91.5]
y=[89.45, 89.45, 89.45, 89.45, 89.45, 89.45,10.55, 10.55, 10.55, 10.55, 10.55, 10.55,
   71.05, 71.05,50., 50.,28.95, 28.95, 50.,50.]
fig,ax=pitch.draw(figsize=(8,16))  
pitch.heatmap_positional(grid_cell, ax=ax, cmap='coolwarm', edgecolors='#22312b')

fig.set_facecolor("black")


#%% Plot Juego de Posición (position game) grid cell with center point
    
path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()]
pitch = mplsoccer.pitch.Pitch(pitch_type="opta",pitch_color="black",line_color="white")

# draw
fig,ax=pitch.draw(figsize=(8,16))
pitch.heatmap_positional(grid_cell, ax=ax, cmap='coolwarm', edgecolors='#22312b')
pitch.scatter(x,y, c='white', s=2, ax=ax)


fig.set_facecolor("black")






