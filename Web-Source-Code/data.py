import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import altair as alt
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt

def app():
    st.title('Data')

    st.subheader('**Data Exploration of Dataset Feature Based Predictions.**')

    st.write('''The dataset for this perdictions was found from Kaggle. It is
    having two datasets separately for traing and testing, In the training set it is
    consist of 576 rows and in the testing set is have 120 rows with 12 columns in each set.''')

    data = pd.read_csv('F:/Data/Experimental/train.csv')
    data1 = pd.read_csv('F:/Data/Experimental/test.csv')

    Y_Train = data.fake
    X_Train = data.drop(columns = 'fake')

    Y_Test = data1.fake
    X_Test = data1.drop(columns = 'fake')

    st.subheader("Features in the Dataset")
    st.write(pd.DataFrame({
        'Feature':['Profile Pic', 'Ratio of UserName length', 'Words in FullName',
        'Ratio of FullName length', 'UserName == FullName','Description Length',
        'External URL','Account Private', 'Number of Posts','Number of followers',
        'Number of followings'],
    }))

    train, test = st.beta_columns(2)
    train.markdown("""
    ### Target Label Balancing - Training
    """)
    train.image('TraindataBalanced.png')

    test.markdown("""### Target Label Balancing - Testing""")
    test.image('TestdataBalanced.png')

    st.sidebar.subheader("Shapes of the DataFrame")
    st.sidebar.write('Training Features Shape:')
    st.sidebar.write(X_Train.shape)
    st.sidebar.write('Training Labels Shape:')
    st.sidebar.write(Y_Train.shape)
    st.sidebar.write('Testing Features Shape:')
    st.sidebar.write(X_Test.shape)
    st.sidebar.write('Testing Labels Shape:')
    st.sidebar.write(Y_Test.shape)

    st.subheader('Correlation Plot')
    st.image('corelation.png')

    st.subheader('Feature Evaluation Compared to Target Variable')
    img1, img2 = st.beta_columns(2)
    img1.image('FollowersVsFake.png',caption='Number of Followers Vs Target')
    img2.image('FollowsVsFake.png',caption='Number of Follows Vs Target')

    img3, img4 = st.beta_columns(2)
    img3.image('FlengthVsFake.png',caption='FullName Length Vs Target')
    img4.image('UlengthVsFake.png',caption='UserName Length Vs Target')

    img5, img6 = st.beta_columns(2)
    img5.image('lengthdescVsFake.png',caption='Description Length Vs Target')
    img6.image('postVsFake.png',caption='Number of Posts Vs Target')

    img7, img8 = st.beta_columns(2)
    img7.image('ProfileVsFake.png',caption='Profile Pic Visibility Vs Target')
    img8.image('unVsfake.png',caption='User Name words Vs Target')

    img9, img10 = st.beta_columns(2)
    img9.image('privateLVsfake.png',caption='Account Private Vs Target')
    img10.image('eURLVsfake.png',caption='External URL Vs Target')


    st.subheader('**Data Exploration of Dataset Text Based Predictions.**')

    st.write('''Data was downloded from Kaggle and Initially it had 5572 rows
     with five columns. After class balncing and data preprocessing now it consist
     of 1494 rows with two columns.''')

    img11, img12 = st.beta_columns(2)
    img11.image('spamcount.png',caption='Dataset Target Before Class Balancing')
    img12.image('balspamcount.png',caption='Dataset Target After Class Balancing')
