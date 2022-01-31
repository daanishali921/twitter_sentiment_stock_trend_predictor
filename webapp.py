import streamlit as st
import pickle
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import metrics
import seaborn as sns

sns.set()

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
                                    options=('Select Ticker', 'GME', 'TSLA', 'AAPL'))
st.sidebar.markdown("***")
daterange_select = st.sidebar.selectbox(label='Choose a date range you would like to be informed about',
                                        options=('Select Range', 'May 24, 2021 - June 24, 2021'))
st.sidebar.markdown("***")


# select_box ticker symbol and date range functionality
def ticker_dataset(ticker_symbol, date_range):
    X, y, trained_model = None, None, None
    if ticker_symbol == 'GME':
        if date_range == 'May 24, 2021 - June 24, 2021':
            X = pd.read_pickle(
                filepath_or_buffer='C:/Users/Daanish/Desktop/capstone_project/project_environment/pickles/latest_GME_demo/X_5-24--6-24_NEW.pkl')
            y = pd.read_pickle(
                filepath_or_buffer='C:/Users/Daanish/Desktop/capstone_project/project_environment/pickles/latest_GME_demo/y_5-24--6-24_NEW_CPT-3.pkl')
            trained_model = pickle.load(open(
                'C:/Users/Daanish/Desktop/capstone_project/project_environment/pickles/latest_GME_demo/pipe_rfc_fit_jan-may_PCT3daylag.pkl',
                'rb'))
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
                                      options=['Retweet Count', 'Likes Count', 'Reply Count', 'Positive Tweet Score',
                                               'Negative Tweet Score', 'Neutral Tweet Score', 'Compound Followers',
                                               'Volume'])


# acf plot functionality
def autocorrelation_chart(ticker_symbol, date_range, feature_select):
    training_data, autocor_plot = None, None
    if ticker_symbol == 'GME':
        if date_range == 'May 24, 2021 - June 24, 2021':
            training_data = pd.read_pickle(
                filepath_or_buffer='C:/Users/Daanish/Desktop/capstone_project/project_environment/pickles/latest_GME_demo/X_1-1--5-21_TRAIN.pkl')
            autocor_plot = sm.graphics.tsa.plot_acf(x=training_data[feature_select],
                                                    lags=15,
                                                    alpha=.05)
            plt.xlabel('Lag (days)')
            plt.ylabel('Autocorrelation')
            plt.title(f'Autocorrelation for {feature_select}')
    elif ticker_symbol == 'TSLA':
        if date_range == 'May 24, 2021 - June 24, 2021':
            pass
    elif ticker_symbol == 'AAPL':
        if date_range == 'May 24, 2021 - June 24, 2021':
            pass
    else:
        return None
    return autocor_plot

# confusion matrix plot functionality
def confusion_plot(X_, y_, model_):
    cf = confusion_matrix(y_true=y_, y_pred=model_.predict(X_))
    ax = sns.heatmap(cf, annot=True, cmap='PuBu')

    ax.set_title('Confusion Matrix\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ax.xaxis.set_ticklabels(['Negative Trend', 'Positive Trend'])
    ax.yaxis.set_ticklabels(['Negative Trend', 'Positive Trend'])

    return plt.show()


def model_evaluation(model, X, y_true, positive_label):
    y_pred = model.predict(X)
    scores = {}
    scores['accuracy'] = round(metrics.accuracy_score(y_true, y_pred), 4)
    scores['precision'] = round(metrics.precision_score(y_true, y_pred, pos_label=positive_label), 4)
    scores['recall'] = round(metrics.recall_score(y_true, y_pred, pos_label=positive_label), 4)
    probs = model.predict_proba(X).T[1]
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, probs, pos_label=positive_label)
    scores['area under precision-recall curve'] = round(metrics.auc(recalls, precisions), 4)
    scores['f1 score'] = round(f1_score(y_true=y_true, y_pred=y_pred, pos_label="Positive_Trend"), 4)
    return scores


st.set_option('deprecation.showPyplotGlobalUse', False)
if stock_select != 'Select Ticker' and daterange_select != 'Select Range':
    with st.expander("The Verdict:"):
        None

    with st.expander(f"Confusion matrix:"):
        y_true = y
        y_pred = model.predict(X)
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        st.pyplot(confusion_plot(X, y, model))
        st.write("Interpreting this matrix:")
        st.write(f'There are {cm[0][0]} and {cm[1][1]} ***correctly*** predicted Negative Trends and Positive Trends respectively.')
        st.write(f'There are {cm[0][1]} and {cm[1][0]} ***incorrectly*** predicted Negative Trends and Positive Trends respectively')

    with st.expander(f"Autocorrelation of {feature_select}:"):
        st.pyplot(autocorrelation_chart(stock_select, daterange_select, feature_select))


