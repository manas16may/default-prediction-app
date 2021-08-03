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
def predict_default(LTV_time,rate_time,hpi_time,gdp_time,uer_time,hpi_orig_time):
    result1=feature(uer_time)
    result2=feature1(LTV_time)
    result3=feature2(gdp_time)
    result4=feature3(rate_time)
    result5=feature4(hpi_time)
    result6=feature5(hpi_orig_time)
    result7=score_func(LTV_time,rate_time,hpi_time,gdp_time,uer_time,hpi_orig_time)
    prediction=clf.predict([[result1,result2,result3,result4,result5,result6,result7]])
    return prediction
def score_func(LTV_time,rate_time,hpi_time,gdp_time,uer_time,hpi_orig_time):
    d3=pd.read_csv('filedataset.csv')
    d4=d3.copy()
    l=["balance_time","LTV_time","interest_rate_time","rate_time","hpi_time","gdp_time","uer_time","FICO_orig_time","LTV_orig_time","Interest_Rate_orig_time","hpi_orig_time"]
    mean1=d4['LTV_time'].mean()
    d4['LTV_time'].fillna(value=mean1, inplace=True)
    vanilladf=d4.copy()
    model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.3),max_features=1.0)
    l=np.array([LTV_time,rate_time,hpi_time,gdp_time,uer_time,hpi_orig_time])
    score=model.fit_predict(l.reshape(1,-1))
    return score
def feature(uer_time):
    if uer_time<6:
        return -0.556782
    elif uer_time>=6 and uer_time<9:
        return 0.861585	
    else:
        return 1.425846
def feature1(ltv_time):
    if (ltv_time< 75):
        return -1.312359
    elif ltv_time>=75 and ltv_time<95:
        return -0.156316
    elif ltv_time>=95 and ltv_time<110:
        return 1.163065
    else:
        return 2.058662
def feature2(gdp):
    if gdp<0:
        return 1.558797	
    elif gdp>=0 and gdp<2:
        return 0.572916	
    elif gdp>=2 and gdp<3:
        return -0.796811
    else:
        return -1.159047	
def feature3(rate):
    if rate<4.300000000000002:
        return -0.854916
    elif rate>=4.300000000000002 and rate<4.500000000000002:
        return -0.015457
    elif rate>=4.500000000000002 and rate<4.8000000000000025:
        return 0.595408
    else:
        return 0.246775	 
def feature4(hpi):
    if hpi<174:
        return 1.170499
    elif hpi>=174 and hpi<188:
        return 0.436312
    elif hpi>=188 and hpi<190:
        return -3.403909
    else:
        return -0.613944
def feature5(hpi1):
    if hpi1<200:
        return -0.946319	
    elif hpi1>=200 and hpi1<210:
        return -0.127332	
    else:
        return 	0.597692
def baseline(LTV_time,rate_time,hpi_time,gdp_time,uer_time,hpi_orig_time):
    result1=feature(uer_time)
    result2=feature1(LTV_time)
    result3=feature2(gdp_time)
    result4=feature3(rate_time)
    result5=feature4(hpi_time)
    result6=feature5(hpi_orig_time)
    result7= score_func(LTV_time,rate_time,hpi_time,gdp_time,uer_time,hpi_orig_time)
   #l=np.array([0.6354436,0.11978002,0.7314715,0.26793107,-0.14555972,0.20253254,0.26342889])
   #l1=np.array([result2,result4,result5,result3,result1,result6,result7])
    z=0.6354436*result2 + 0.11978002*result4 + 0.7314715*result5 + 0.26793107*result3 -0.14555972*result1+0.20253254*result6+0.26342889*result7
    l=1/(1+math.exp(-z))
    return round(l,2)
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
    rate_time = st.text_input("rate_time","Type Here")
    hpi_time = st.text_input("hpi_time","Type Here")
    gdp_time = st.text_input("gdp_time","Type Here")
    uer_time=  st.text_input("uer_time","Type Here")
    hpi_orig_time=st.text_input("hpi_orig_time","Type Here")
    result=""
    if st.button("Logistic Baseline"):
        result1=baseline(float(Ltv_time),float(rate_time),float(hpi_time),float(gdp_time),float(uer_time),float(hpi_orig_time))
        myfunction2= np.vectorize
        st.success('The probability of default is {}'.format(result1))
    if st.button("Predict default by light gbm"):
        result=predict_default(float(Ltv_time),float(rate_time),float(hpi_time),float(gdp_time),float(uer_time),float(hpi_orig_time))
        if result[0]==0:
           st.success('It is a non default case')
        else:
           st.success('It is a default case') 
    if st.button("About"):
        st.text("Built with Streamlit")
if __name__=='__main__':
    main()
