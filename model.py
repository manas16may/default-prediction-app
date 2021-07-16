# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 13:14:52 2021

@author: manas
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
d3=pd.read_csv('C:/Users/manas/Downloads/filename1.csv')
d4=d3.drop(['Unnamed: 0'],axis=1)
d4.drop(['balance_orig_time'],inplace=True,axis=1)
l=["balance_time","LTV_time","interest_rate_time","rate_time","hpi_time","gdp_time","uer_time","FICO_orig_time","LTV_orig_time","Interest_Rate_orig_time","hpi_orig_time"]
mean1=d4['LTV_time'].mean()
d4['LTV_time'].fillna(value=mean1, inplace=True)
vanilladf=d4.copy()
luer=[]
for i in vanilladf['uer_time']:
  if i<6:
    luer.append(1)
  else:
    luer.append(0)
vanilladf['feature_uer']=luer
from sklearn.ensemble import IsolationForest
model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.5),max_features=1.0)
model.fit(vanilladf[['uer_time']])
vanilladf['score']=model.decision_function(vanilladf[['uer_time']])
mean_value=vanilladf['LTV_time'].mean()
vanilladf['LTV_time'].fillna(value=mean_value, inplace=True)
vanilladf['bins']=pd.cut(x=vanilladf['LTV_time'],bins=[0,80,650],labels=[0,1])
vanilladf['bins']=vanilladf['bins'].fillna(vanilladf['bins'].mode()[0])
X1=vanilladf[['LTV_time', 'interest_rate_time', 'hpi_time', 'gdp_time', 'uer_time','feature_uer','bins','score']]
y1=vanilladf.iloc[:,-1]
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=42,stratify=y1)
m2=LogisticRegression(max_iter=500)
m2.fit(X_train1,y_train1)
pickle_out=open("creditrisk.pkl",'wb')
pickle.dump(m2,pickle_out)
pickle_out.close()
