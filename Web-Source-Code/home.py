import streamlit as st
import pandas as pd
from PIL import Image
import datetime
import matplotlib.pyplot as plt

def app():

    image = Image.open('home.jpg')
    st.sidebar.image(image)
    st.sidebar.image('image1.png')

    today = datetime.date.today()
    st.sidebar.date_input('Date :', today)

    st.title('Home')

    st.subheader('Introduction')

    st.write("""This **User Prediction** web application was develop to predict and
    detect genuine users and fake users in Instagram and also it facilitate to detect
    whether the input message is a genuine message or a fake message. Initially a training set of data is
    fed into the model and train the model according to the label (target variable).
    The Machine Learning models, **Logistic Regression**, **K-Nearest Neighbors**,
    **Support Vector Machine** and **Naive Bayes** are applied and then test the model
    by passing parameter values and text messages. As shown in the following two tables Logistic Regression
    model gives the best accuracy. Apart from that text based prediction shows the best accuracy.""")

    st.subheader('Objectives')

    st.markdown(
    """
     * Training model Using Machine Learning Algorithms.
     * Predict Genuine and Fake users for Features of the user.
     * Predict Genuine and Fake Messages to input message.
     * Testing is done using features and text messages.

    """)
    st.subheader('Accuracy of Feature Based Predicting ML Models')

    st.write(pd.DataFrame({
        'Learning Algorithm':['Logisti Regression','K-Nearest Neighbors','Naive Bayes'],
        'Accuracy (%)':[91 ,87, 53 ],
    }))

    st.subheader('Accuracy of Text Based Predicting ML Models')

    st.write(pd.DataFrame({
        'Learning Algorithm':['Logistic Regression','Naive Bayes','K-Nearest Neighbors'],
        'Accuracy (%)':[96, 97, 69 ],
    }))

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'LR', 'KNN', 'NB'
    sizes = [91, 87, 53]

    labels1 = 'LR','NB','KNN'
    sizes1 = [96,97,69]

    fig = plt.figure(figsize=(10,3))
    plt.subplot(1, 2, 1)
    plt.bar(labels,sizes, color = 'green',width = 0.25)
    plt.title("Feaure Prediction")

    plt.subplot(1, 2, 2)
    plt.bar(labels1,sizes1, color = 'blue',width = 0.25)
    plt.title("Text Detection")

    st.pyplot(fig)
