import streamlit as st
import pickle
import pandas as pd

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

X_march = pd.read_pickle(filepath_or_buffer='C:/Users/Daanish/Desktop/capstone_project/project_environment/pickles/evaluate/X_3-1--3-31.pkl')
y_march = pd.read_pickle(filepath_or_buffer='C:/Users/Daanish/Desktop/capstone_project/project_environment/pickles/evaluate/y_3-1--3-31.pkl')

st.write("""
# Twitter Stock Trend Informer

This app will inform the user of how much twitter trends affect stock trends for certain stocks
""")

st.sidebar.header('Stock and Date choices:')

stock_option = st.sidebar.selectbox(
    label='Which stock would you like to be informed about?',
    options=('GME', 'stock2', 'stock3')

)
st.sidebar.markdown("***")
daterange_option = st.sidebar.selectbox(
    label='Choose a date range you would like to be informed about',
    options=('range1', 'range2', 'range3')
)

model = pickle.load(open('C:/Users/Daanish/Desktop/capstone_project/project_environment/pickles/training/pipe_rfc_fit_jan-feb_3daylag.pkl', 'rb'))

prediction_march = model.predict(X_march)

st.header('Prediction of March 2021')
st.write(prediction_march)
st.write('---')
