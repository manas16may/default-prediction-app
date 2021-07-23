# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 13:14:52 2021

@author: manas
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
d3=pd.read_csv('C:/Users/manas/Downloads/train.csv')
#balance_orig and balance_obs are correlated
#hpi_time and uer_time are highly negatively correlated
l=["balance_time","LTV_time","interest_rate_time","rate_time","hpi_time","gdp_time","uer_time","FICO_orig_time","LTV_orig_time","Interest_Rate_orig_time","hpi_orig_time"]
"""###Data split for simple logistic"""

mean1=d3['LTV_time'].mean()
d3['LTV_time'].fillna(value=mean1, inplace=True)

X=d3[[ 'balance_time', 'LTV_time', 'interest_rate_time', 'rate_time','hpi_time', 'gdp_time', 'uer_time', 'REtype_CO_orig_time','REtype_PU_orig_time', 'REtype_SF_orig_time', 'investor_orig_time', 'FICO_orig_time', 'LTV_orig_time','Interest_Rate_orig_time', 'hpi_orig_time']]
y=d3.iloc[:,-1]
m1=LogisticRegression(max_iter=1000)
m1.fit(X,y)
d4=pd.read_csv('C:/Users/manas/Downloads/test.csv')
X1=d4[[ 'balance_time', 'LTV_time', 'interest_rate_time', 'rate_time','hpi_time', 'gdp_time', 'uer_time', 'REtype_CO_orig_time','REtype_PU_orig_time', 'REtype_SF_orig_time', 'investor_orig_time', 'FICO_orig_time', 'LTV_orig_time','Interest_Rate_orig_time', 'hpi_orig_time']]
y1=d4.iloc[:,-1]
y=m1.predict(X1)
vanilladf=d3.copy()
from sklearn.ensemble import IsolationForest
model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.5),max_features=1.0)
model.fit(vanilladf[['uer_time']])

vanilladf['score']=model.decision_function(vanilladf[['uer_time']])
#X_test1['score']=model.decision_function(X_test1[['uer_time']])

d4['score']=model.decision_function(d4[['uer_time']])
import scorecardpy as sc

bins = sc.woebin(d3, y="status_time")

l=[ 'LTV_time', 'interest_rate_time']
data_woe = sc.woebin_ply(d3, bins)
data_woe.drop(['id_woe'],inplace=True,axis=1)
test_woe=sc.woebin_ply(X1, bins)


X2=data_woe.iloc[:,1:16]
y2=data_woe.iloc[:,0]

p={}
for i in l:
     p[i]=bins[i]['total_iv'][0]
useful=[]
for i in p:
  if(p[i]>=0.2):
    useful.append(i)
useful.extend(['score'])
print(useful)
comparedf=pd.DataFrame()
L=['LTV_time', 'interest_rate_time','uer_time']
for i in useful:
  if(i!='score' and i!='hpi_time' and i!='gdp_time' ):
    comparedf[i+'_woe']=data_woe[i+'_woe']
comparedf['uer_time_woe']=vanilladf['uer_time']
comparedf['score']=vanilladf['score']
comparedf['gdp_time']=vanilladf['gdp_time']
comparedf['hpi_time']=vanilladf['hpi_time']
comparedf['status_time']=vanilladf['status_time']
print (comparedf.shape)
X4=comparedf.loc[:,comparedf.columns != 'status_time']
y4=comparedf.loc[:,'status_time']
d4['LTV_time_woe']=test_woe['LTV_time_woe']
d4['uer_time_woe']=test_woe['uer_time_woe']
d4['interest_rate_time_woe']=test_woe['interest_rate_time_woe']
m5=LogisticRegression(max_iter=500,solver='saga')
m5.fit(X4,y4)
xtest=d4[['LTV_time_woe', 'interest_rate_time_woe', 'hpi_time', 'gdp_time', 'uer_time_woe', 'score']]
y5=m5.predict(xtest)
from lightgbm import LGBMClassifier
model1 = LGBMClassifier(max_depth=1)
model1.fit(X4, y4)
y8=model1.predict(xtest)
pickle_out=open("creditrisk.pkl",'wb')
pickle.dump(model1,pickle_out)
pickle_out.close()
