import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error, rand_score


# make containers
header = st. container()
data_sets= st. container()
featuress = st. container()
model_training = st. container()
with header:
    st.title("kashti ki app ")
    st.text("in the projects we will work on kashti data")
    
    
with data_sets:
    st.header("kashti doob gay ,hhhwww!")
    st.text("we work with titanic dataset") 
    # imort data
    df = sns.load_dataset('titanic')
    df =df.dropna()
    st.write(df.head(10))
    st.subheader("samba ,are ooohh sambha kitny admi thy?")
    st.bar_chart(df['sex'].value_counts())
    
    # other plot
    st.subheader("class ka hisab sy farak")
    st.bar_chart(df['class']. value_counts())
    # bar plot age
    st.bar_chart(df['age'].sample(10)) # or .head(10)
    
    
    
    
with featuress:
    st.header("there are our app features")         
    st.text("away bhot sary freature add karty hay")
    st.markdown('1.feature       1:  this will tell us pata nahi? ')
    st.markdown('2.feature       2:  this will tell us pata nahi? ')
    st.markdown('3.feature       3:  this will tell us pata nahi? ')
    
    
    
    
    
with model_training:
    st.header("kashti walo ka kaya bana?-model training")
    st.text("is may apnay parameter set kar sakty hay")                                                                                                                       
    
    # making columns
    input ,display = st.columns(2)
    
    # pehlay column main ap k selection points hun
    max_depth =input.slider ("how many poeple do you know? ", min_value=10 , max_value=100,value =20 ,step= 5)
    
    
    # n_estimators
    n_estimators = input.selectbox("how many tree should be there in a RF",options=[50,100,200,300,'NO LIMIT'])
    
    # adding list of features
    input.write(df.columns)
    # input features from user
    input_features = input.text_input('which freature we shoud use?')
    
    
    # machine learnig model
    model = RandomForestRegressor(max_depth=max_depth ,n_estimators=n_estimators)
    # yaha par ham aik condation lagayen gya
    if n_estimators =='no limit':
         model  = RandomForestRegressor(max_depth=max_depth)
    else: 
        model   = RandomForestRegressor(max_depth=max_depth ,n_estimators=n_estimators)
    
    
    # difene x and y
    x = df[[input_features]]
    y = df[['fare']]
    
    # fit our model
    model.fit(x,y)
    pred = model.predict(y)
    
  # disply metriess

display.subheader("mean absolute error of the model is:")
display.write(mean_absolute_error(y,pred))
display.subheader("mean squared error of the model is:")
display.write(mean_squared_error(y,pred))
display.subheader("R squared score error of the model is:")
display.write(r2_score(y,pred))