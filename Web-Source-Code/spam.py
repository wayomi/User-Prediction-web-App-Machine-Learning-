import streamlit as st
import pandas as pd
import numpy as np
import nltk
#nltk.download("all")
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

def app():

    st.title('Text Based Prediction')

    st.write('''This prediction model results whether the message or the text
    partition the user entered is a Genuine message or a Fake message. In order
    to choose the best fitted model, trained the dataset using Logistic Regression,
    KNN and Naive Bayes. After testing the model, conclude that Logit Regression and
    Naive Bayes results the predictions with a accuracy of 97% ''')

    data = pd.read_csv('F:/Data/Experimental/Balancespam.csv', encoding = 'latin-1')

    df_data = data[["text","fake"]]

    df_x = df_data['text']
    df_y = df_data['fake']

    #corpus = df_x
    #cv = CountVectorizer()
    #X = cv.fit_transform(corpus)

    vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")
    features = vec.fit_transform(data["text"])
    X = vec.fit_transform(df_x)

    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)

    st.subheader("Enter Text Here..")
    message = st.text_input("")
    #all_ml_models = (["Logistic Regression","Random Forest","NB","KNN"])

    st.subheader('Classifiers')
    st.markdown(
    """
     * LR ~ Logistic Regression
     * NB ~ Naive Bayes
     * KNN ~ K-Nearest Neighbors.
    """)

    st.subheader("Select the Classifier")
    model_choice = st.selectbox('',("LR","NB","KNN"))

    if st.button("Click to Predict"):
        vect = vec.transform([message]).toarray()
        if model_choice == 'LR':
            logreg=LogisticRegression(class_weight = 'balanced')
            logreg.fit(X_train,y_train)

            if (logreg.predict(vect) == "spam"):
                st.success("Fake Message")
            else:
                st.success("Genuine Message")

        elif model_choice == 'NB':
            clf = MultinomialNB()
            clf.fit(X_train,y_train)

            if (clf.predict(vect) == "spam"):
                st.success("Fake Message")
            else:
                st.success("Genuine Message")

        elif model_choice == 'KNN':
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train,y_train)

            if (knn.predict(vect) == "spam"):
                st.success("Fake Message")
            else:
                st.success("Genuine Message")
