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

pickle_in=open("C:/Users/manas/creditapp/creditrisk.pkl","rb")
clf=pickle.load(pickle_in)
def predict_default(LTV_time,interest_rate_time,hpi_time,gdp_time,uer_time):
    result1=feature(uer_time)
    result2=featureb(LTV_time)
    result3=score_func(uer_time)
    prediction=clf.predict([[LTV_time,interest_rate_time,hpi_time,gdp_time,uer_time,result1,result2,result3]])
    return prediction
def score_func(uer_time):
    d3=pd.read_csv('C:/Users/manas/Downloads/filename1.csv')
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
        return 1
    else:
        return 0
def featureb(ltv_time):
    if (ltv_time<=80):
        return 0
    else:
        return 1
def main():
    st.title("Bank Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Ltv_time = st.text_input("Ltv_time","Type Here")
    interest_rate_time = st.text_input("interest_rate_time","Type Here")
    hpi_time = st.text_input("hpi_time","Type Here")
    gdp_time = st.text_input("gdp_time","Type Here")
    uer_time=  st.text_input("uer_time","Type Here")
    result=""
    if st.button("Predict default"):
        result=predict_default(int(Ltv_time),int(interest_rate_time),int(hpi_time),int(gdp_time),int(uer_time))
        if result[0]==0:
           st.success('It is a non default case')
        else:
           st.success('It is a default case') 
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")
if __name__=='__main__':
    main()