import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

def app():
    st.title('Feature Based Prediction' )

    st.write('''This app predicts the Genuine users (0) and Fake users (1)
    in Instagram using Logistic Regression, KNN and Naive Bayes.
    The best predictions is given by the Logistic Regression model which scores a
    91% of accuracy. The Probability of predicting the users is visualize in the
    in prediction Probability section. ''')

    st.sidebar.header('User Input Parameters')

    def user_input_features():

        profilepic = st.sidebar.radio('Is there a Profile Pic (1-Yes, 0-No)', [0,1])
        lengthusername = st.sidebar.slider('Length of the username', 0.0,0.89)
        fullnamewords = st.sidebar.slider('Number of words in FullName', 1.0, 9.0)
        lengthfullname = st.sidebar.slider('Length of the Fullname', 0.0, 1.0)
        nameusername = st.sidebar.radio('Is UserName equals FullName', [0,1])
        #nameusername = st.sidebar.selectbox('UserName == FullName', ("1","0"))
        descriptionlength = st.sidebar.slider('Length of the description', 0.0,149.0)
        externalURL = st.sidebar.radio('Is there a External URL (1-Yes, 0-No)', [0,1])
        private = st.sidebar.radio('Is account Private (1-Yes, 0-No)', [0,1])
        posts = st.sidebar.slider('Number of Posts', 0.0, 1879.0)
        followers = st.sidebar.slider('Number of followers', 0.0,4000.0)
        follows = st.sidebar.slider('Number of followings', 0.0, 7455.0)
        data = {'Profile Pic': profilepic,
                'Ratio of UserName length': lengthusername,
                'Words in FullName': fullnamewords,
                'Ratio of FullName length': lengthfullname,
                'UserName == FullName': nameusername,
                'Description Length': descriptionlength,
                'External URL': externalURL,
                'Account Private': private,
                'Number of Posts': posts,
                'Number of followers': followers,
                'Number of followings': follows}
        features = pd.DataFrame(data, index=[0])
        return features

    features = user_input_features()

    st.subheader('User Input parameters')
    st.write(features)

    data = pd.read_csv('F:/Data/Experimental/train.csv')

    Y_Train = data.fake
    X_Train = data.drop(columns = 'fake')

    st.subheader('Classifiers')
    st.markdown(
    """
     * LR ~ Logistic Regression
     * KNN ~ K-Nearest Neighbors.
     * NB ~ Naive Bayes

    """)

    st.subheader("Select the Classifier")
    model_choice = st.selectbox('',("LR","KNN","NB"))

    if st.button("Click to Predict"):
        if model_choice == 'LR':
            logreg=LogisticRegression(class_weight = 'balanced')
            model1 = logreg.fit(X_Train,Y_Train)

            if (model1.predict(features) == 1):
                st.success("Fake User")
            else:
                st.success("Genuine User")

        elif model_choice == 'KNN':
            knn = KNeighborsClassifier(n_neighbors=10)
            model3 = knn.fit(X_Train,Y_Train)

            if (model3.predict(features) == 1):
                st.success("Fake User")
            else:
                st.success("Genuine User")

        elif model_choice == 'NB':
            clf = MultinomialNB()
            model4 = clf.fit(X_Train,Y_Train)

            if (model4.predict(features) == 1):
                st.success("Fake User")
            else:
                st.success("Genuine User")
