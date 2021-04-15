import streamlit as st
from multiapp import MultiApp
from apps import home, data, userinput, spam

app = MultiApp()

st.markdown("""
# Genuine | Fake User Prediction in instagram - Web App
#### By Wayomi S Jayantha
""")

app.add_app("Home", home.app)
app.add_app("Data", data.app)
#app.add_app("Logistic Regression", lr.app)
#app.add_app("Neural Network Prediction", nntext.app)
app.add_app("Feature Based Prediction", userinput.app)
app.add_app("Text Based Prediction", spam.app)

app.run()
