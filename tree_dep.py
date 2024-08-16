#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
pickle_in=open('decisiontree.pkl','rb')
clf=pickle.load(pickle_in)


# In[2]:


# -*- coding: utf-8 -*-
"""
Created on Fri 09-08 13:01:17 2024

"""

import pandas as pd
import streamlit as st 
from sklearn.tree import DecisionTreeClassifier

st.title('Model Deployment: Decision Tree')

st.sidebar.header('User Input Parameters')

def user_input_features():
    
    age= st.sidebar.number_input("Insert the age")
    thalch = st.sidebar.number_input("maximum heart rate achieved")
    oldpeak = st.sidebar.number_input("Insert the ST depression value") 
    data = {'age':age,
            'thalch':thalch,
            'oldpeak':oldpeak}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

data_heart = pd.read_excel(r"C:\Users\nitin\Downloads\Latest DS material\DecisionTree-Streamlit\heart_disease.xlsx",sheet_name= 'Heart_disease')
data_heart = data_heart.dropna()

X = data_heart[['age','thalch','oldpeak']]
Y = data_heart[['num']]
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('class 0' if prediction_proba[0][1] > 0.5 else 'class 1')

st.subheader('Prediction Probability')
st.write(prediction_proba)

