import streamlit as st
import pickle
import pandas as pd
import statsmodels.api as sm

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.write("""
# Twitter Stock Trend Informer
This app will inform the user of how much twitter trends affect stock trends for certain stocks
""")

st.sidebar.header('Stock and Date choices:')
stock_select = st.sidebar.selectbox(label='Choose the stock ticker',
                                    options=('Select Ticker','GME', 'TSLA', 'AAPL'))
st.sidebar.markdown("***")
daterange_select = st.sidebar.selectbox(label='Choose a date range you would like to be informed about',
                                        options=('Select Range', 'May 24, 2021 - June 24, 2021'))
st.sidebar.markdown("***")

# select_box ticker symbol and date range functionality
def ticker_dataset(ticker_symbol, date_range):
    X, y, trained_model = None, None, None
    if ticker_symbol == 'GME':
        if date_range == 'May 24, 2021 - June 24, 2021':
            X = pd.read_pickle(filepath_or_buffer='C:/Users/Daanish/Desktop/capstone_project/project_environment/pickles/latest_GME_demo/X_5-24--6-24_NEW.pkl')
            y = pd.read_pickle(filepath_or_buffer='C:/Users/Daanish/Desktop/capstone_project/project_environment/pickles/latest_GME_demo/y_5-24--6-24_NEW_CPT-3.pkl')
            trained_model = pickle.load(open('C:/Users/Daanish/Desktop/capstone_project/project_environment/pickles/latest_GME_demo/pipe_rfc_fit_jan-may_PCT3daylag.pkl', 'rb'))
    elif ticker_symbol == 'TSLA':
        if date_range == 'May 24, 2021 - June 24, 2021':
            pass
    elif ticker_symbol == 'AAPL':
        if date_range == 'May 24, 2021 - June 24, 2021':
            pass
    else:
        pass
    return X, y, trained_model
X, y, model = ticker_dataset(stock_select, daterange_select)


feature_select = st.sidebar.selectbox(label='Choose features to view their autocorrelation graph',
                                      options=['Negative Tweet Score', 'Positive Tweet Score', 'Compound Tweet Score'])

# acf plot functionality
def autocorrelation_chart(ticker_symbol, date_range, feature_select):
    training_data, autocor_plot = None, None
    if ticker_symbol == 'GME':
        if date_range == 'May 24, 2021 - June 24, 2021':
            training_data = pd.read_pickle(filepath_or_buffer='C:/Users/Daanish/Desktop/capstone_project/project_environment/pickles/latest_GME_demo/X_1-1--5-21_TRAIN.pkl')
            autocor_plot = sm.graphics.tsa.plot_acf(x=training_data[feature_select],
                                                    lags=15,
                                                    alpha=.05)
    elif ticker_symbol == 'TSLA':
        if date_range == 'May 24, 2021 - June 24, 2021':
            pass
    elif ticker_symbol == 'AAPL':
        if date_range == 'May 24, 2021 - June 24, 2021':
            pass
    else:
        return None
    return autocor_plot
if stock_select != 'Select Ticker' and daterange_select != 'Select Range':
    st.pyplot(autocorrelation_chart(stock_select, daterange_select, feature_select))

# score_select = st.sidebar.



# model = pickle.load(open('C:/Users/Daanish/Desktop/capstone_project/project_environment/pickles/training/pipe_rfc_fit_jan-feb_3daylag.pkl', 'rb'))
#
# prediction_march = model.predict(X_march)
#
# st.header('Features of Mach 2021')
# st.write(X_march[['Retweet Count', 'Likes Count', 'Reply Count',
#                                   'Positive Tweet Score', 'Negative Tweet Score', 'Neutral Tweet Score',
#                                   'Compound Followers', 'Volume']])
# st.write('---')
#
# st.header('March 2021 True Labels')
# st.write(y_march)

