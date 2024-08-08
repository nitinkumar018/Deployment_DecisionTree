#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
pickle_in=open('dtree.pkl','rb')
model_tree=pickle.load(pickle_in)


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
    
    trestbps= st.sidebar.number_input("Insert the blood pressure")
    chol = st.sidebar.number_input("Insert the cholesterol measure")
    oldpeak = st.sidebar.number_input("Insert the ST depression value") 
    data = {'trestbps':trestbps,
            'chol':chol,
            'oldpeak':oldpeak}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

data_heart = pd.read_excel(r"C:\Users\nitin\Downloads\Latest DS material\DecisionTree-Streamlit\heart_disease.xlsx",sheet_name= 'Heart_disease')
data_heart.drop(["age"],inplace=True,axis = 1)
data_heart.drop(["sex"],inplace=True,axis = 1)
data_heart.drop(["cp"],inplace=True,axis = 1)

data_heart.drop(["fbs"],inplace=True,axis = 1)
data_heart.drop(["restecg"],inplace=True,axis = 1)
data_heart.drop(["thalch"],inplace=True,axis = 1)

data_heart.drop(["exang"],inplace=True,axis = 1)
data_heart.drop(["slope"],inplace=True,axis = 1)
data_heart.drop(["thal"],inplace=True,axis = 1)
data_heart = data_heart.dropna()

X = data_heart.iloc[:,[0,1,2]]
Y = data_heart.iloc[:,-1]
model_tree = DecisionTreeClassifier(criterion='entropy')
model_tree.fit(X,Y)

prediction = model_tree.predict(df)
prediction_proba = model_tree.predict_proba(df)

st.subheader('Predicted Result')
st.write('class 4' if prediction_proba[0][1] > 0.5 else 'class 2')

st.subheader('Prediction Probability')
st.write(prediction_proba)


# In[ ]:




