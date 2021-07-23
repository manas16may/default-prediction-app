# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 13:26:57 2021

@author: manas
"""


import pandas as pd
import pickle
import streamlit as st
from sklearn.ensemble import IsolationForest
from PIL import Image
import math 
import numpy as np
pickle_in=open("creditrisk.pkl","rb")
clf=pickle.load(pickle_in)
def predict_default(LTV_time,interest_rate_time,hpi_time,gdp_time,uer_time):
    result1=feature(uer_time)
    result2=feature1(LTV_time)
    result3=feature2(interest_rate_time)
    result4=score_func(uer_time)
    prediction=clf.predict([[result1,result2,result3,hpi_time,gdp_time,result4]])
    return prediction
def score_func(uer_time):
    d3=pd.read_csv('C:/Users/manas/Downloads/train.csv')
    d4=d3.copy()
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
    model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.5),max_features=1.0)
    model.fit(vanilladf[['uer_time']])
    score=model.decision_function([[uer_time]])
    return score
def feature(uer_time):
    if uer_time<6:
        return -2.573210
    elif uer_time>=6 and uer_time<8:
        return 2.897169	
    else:
        return 4.199933	
def feature1(ltv_time):
    if (ltv_time< 85):
        return -1.370708
    elif ltv_time>=85 and ltv_time<100:
        return -0.256733
    elif ltv_time>=100 and ltv_time<110:
        return 1.463473
    else:
        return 3.286999
def feature2(hpi):
    if hpi<3:
        return -1.668817
    elif hpi>=3 and hpi<7:
        return 0.189833
    else:
        return 	0.533799
def baseline(LTV_time,interest_rate_time,hpi_time,gdp_time,uer_time):
    result1=feature(uer_time)
    result2=feature1(LTV_time)
    result3=feature2(interest_rate_time)
    result4=score_func(uer_time)
    l=np.array([ 0.28058067,0.43214724,0.88450645,-0.95642009,-1.88211948,0.02419201])
    l1=np.array([result2,result3,hpi_time,gdp_time,result1,result4])
    eq=np.dot(l,l1)
    z=1/(1+math.exp(-eq))
    return round(z,2)
def main():
    st.title("Credit risk")
    html_temp = """
    <div style="background-color:orange;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Credit Risk ML App </h2>
    </div>
    """
    st.header('This app is created to predict if the customer will default after taking loan or not')
    st.markdown(html_temp,unsafe_allow_html=True)
    Ltv_time = st.text_input("Ltv_time","Type Here")
    interest_rate_time = st.text_input("interest_rate_time","Type Here")
    hpi_time = st.text_input("hpi_time","Type Here")
    gdp_time = st.text_input("gdp_time","Type Here")
    uer_time=  st.text_input("uer_time","Type Here")
    result=""
    if st.button("Baseline"):
        result1=baseline(float(Ltv_time),float(interest_rate_time),float(hpi_time),float(gdp_time),float(uer_time))
        
        st.success('The probability of default is {}'.format(result1))
    if st.button("Predict default"):
        result=predict_default(float(Ltv_time),float(interest_rate_time),float(hpi_time),float(gdp_time),float(uer_time))
        if result[0]==0:
           st.success('It is a non default case')
        else:
           st.success('It is a default case') 
    if st.button("About"):
        st.text("Built with Streamlit")
if __name__=='__main__':
    main()
