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
            footer:after 
                {
                content:'by Daanish Ali'; 
                visibility: visible;
                display: block;
                position: relative;
                padding: 5px;
                top: 2px;
                }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


st.sidebar.header('Stock and Date choices:')
stock_select = st.sidebar.selectbox(label='Choose the stock ticker',
                                    options=('Select Ticker', 'GME'))
daterange_select = st.sidebar.selectbox(label='Choose a date range you would like to be informed about',
                                        options=('Select Range', 'May 26, 2021 - June 24, 2021'))
st.sidebar.markdown("***")


st.write(f"""
# Twitter Stock Trend Informer 
###### This app will inform the user of how much twitter trends affect stock trends for certain stocks \n 

Our goal is to predict the trend (*Positive or Negative*) of stock prices 3 days in advance within the selected date range using a **Random Forest Binary Classifier**.
We then build an experiment involving two strategies: \n
1. Buy stock on first day of the date range and then hold onto the stock --> **Record gains/losses**
2. Buy stock on first day of the date range and then be informed by the predictive model on when to buy or sell --> **Record gains/losses** \n
After choosing a stock ticker and date range, the results of our experiment and information about the features/model will appear below:
""")



# select_box ticker symbol and date range functionality

def ticker_data(ticker_symbol, date_range):
    X, y, trained_model, df_experiment_predictions, verdict_text = None, None, None, None, None
    if ticker_symbol == 'GME':
        if date_range == 'May 26, 2021 - June 24, 2021':
            X = pd.read_pickle(filepath_or_buffer='X_5-24--6-24_.pkl')
            y = pd.read_pickle(filepath_or_buffer='y_5-24--6-24_CPT-3.pkl')
            trained_model = pickle.load(open('pipe_rfc_fit_jan-may_PCT3daylag.pkl', 'rb'))
            df_experiment_predictions = pd.read_pickle(filepath_or_buffer='df_gme_prediction_exp_1.pkl')
            verdict_text = 'What the model is doing is predicting stock trends ***three days in the future***. We use this information to build an experiment;\n' \
                           'We use the predictions the model has given us to suggest a buy or sell.\n' \
                           'This experiment compares two strategies: buy and hold vs. the suggestions from the model.\n' \
                           'The graph presents the results from both strategies. \n \n' \
                           'Both strategies start with $350 and we buy one GME stock. \n \n'  \
                           '**The way the predictive model strategy is set up is the following:** \n \n' \
                           '1. We start with buying the stock on the first day at opening price. \n \n' \
                           '2. Then, we loop through each day; every iteration does a check whether or not we are currently holding a stock, then, ' \
                           'based on a positive/negative stock trend prediction for the next day, we decide to keep the stock or sell ' \
                           'it at closing (positive = hold/ negative = sell). \n \n' \
                           '3. If there are successive negative trend predictions and we do not have the stock, ' \
                           'we continue to not hold it until a positive prediction occurs. ' \
                           'We then buy at opening price the day of the predicted positve trend (the next day). \n \n \n \n' \
                           'Unfortunately, we can see our models performance within the experiment did not perform better than a baseline buy-and-hold strategy.'
    elif ticker_symbol == 'TSLA':
        if date_range == 'May 24, 2021 - June 24, 2021':
            pass
    else:
        pass
    return X, y, trained_model, df_experiment_predictions, verdict_text


X, y, model, df_experiment, verdict_text = ticker_data(stock_select, daterange_select)

st.sidebar.header('Learn about the features that trained the model:')
feature_select = st.sidebar.selectbox(label='Choose features to view their autocorrelation graph',
                                      options=['Retweet Count', 'Likes Count', 'Reply Count', 'Positive Tweet Score',
                                               'Negative Tweet Score', 'Neutral Tweet Score', 'Compound Followers',
                                               'Volume'])


# acf plot functionality
def autocorrelation_chart(ticker_symbol, date_range, feature_select):
    training_data, autocor_plot = None, None
    if ticker_symbol == 'GME':
        if date_range == 'May 26, 2021 - June 24, 2021':
            training_data = pd.read_pickle(filepath_or_buffer='X_1-1--5-21_TRAIN.pkl')
            autocor_plot = sm.graphics.tsa.plot_acf(x=training_data[feature_select],
                                                    lags=15,
                                                    alpha=.05)
            plt.xlabel('Lag (days)')
            plt.ylabel('Autocorrelation')
            plt.title(f'Autocorrelation for {feature_select}')
    elif ticker_symbol == 'TSLA':
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


def graph_algorithm_model(df):
    prediction_series_ = df['buy/sell preds +2']
    open_series_ = df['Open']
    close_series_ = df['Adj Close']
    stock_hold = True

    iterable = []
    for pred, open__, close__ in zip(prediction_series_, open_series_, close_series_):
        iterable.append((pred, open__, close__))

    holding = 350
    close_previous = iterable[0][2]  # previous day hold price if stock is held
    open_previous = iterable[0][1]  # opening price of first day

    final_list = []
    final_list.append(holding + (close_previous - open_previous))
    holding = holding + (close_previous - open_previous)

    for i, e in enumerate(iterable):
        if stock_hold == True:
            if e[0] == -1.0:
                holding = holding + (iterable[i + 1][2] - close_previous)  # update holdings for end of day price
                final_list.append(holding)  # append updated hold value onto list
                stock_hold = False  # sell stock
                close_previous = iterable[i + 1][2]  # update previously used close price
                continue

            elif e[0] == 1.0:
                holding = holding + (iterable[i + 1][2] - close_previous)
                final_list.append(holding)
                close_previous = iterable[i + 1][2]  # update previously used close price
                continue

        if stock_hold == False:
            if e[0] == -1.0:
                # hold price stays the same
                # not updating previous close price
                final_list.append(holding)  # append same hold value onto list
                continue

            elif e[0] == 1.0:
                stock_hold = True  # buy stock AT OPENING PRICE
                holding = holding + (iterable[i + 1][2] - iterable[i + 1][
                    1])  # Profits (if correct) or losses are from the difference between close and open price
                final_list.append(holding)
                close_previous = iterable[i + 1][2]
                continue
    model_outcome = pd.DataFrame(data=final_list, columns=['model_outcome'])[:21]
    model_outcome.index = prediction_series_.index
    return model_outcome


def graph_algorithm_base(df):
    open_series_ = df['Open']
    close_series_ = df['Adj Close']

    iterable = []
    for open__, close__ in zip(open_series_, close_series_):
        iterable.append((open__, close__))

    holding = 350
    close_previous = iterable[0][1]  # previous day hold price if stock is held
    open_previous = iterable[0][0]  # opening price of first day

    final_list = []
    final_list.append(holding + (close_previous - open_previous))
    holding = holding + (close_previous - open_previous)

    try:
        for i, e in enumerate(iterable):
            holding = holding + (iterable[i + 1][1] - close_previous)  # update holdings for end of day price
            final_list.append(holding)  # append updated hold value onto list
            close_previous = iterable[i + 1][1]  # update previously used close price

    except IndexError:
        buy_hold_outcome = pd.DataFrame(data=final_list, columns=['buy_hold_outcome'])
        buy_hold_outcome.index = open_series_.index
        return buy_hold_outcome


def model_experiment_plot(df):
    base_results = graph_algorithm_base(df_experiment)
    pred_results = graph_algorithm_model(df_experiment)

    plt.plot(base_results, 'r', linewidth=2)
    plt.plot(pred_results, 'b', linestyle=':', linewidth=5)

    plt.xlabel('Date', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylabel('Current Holdings', fontsize=12)

    a, b = 'Buy-and-hold Strategy Gains/Losses', 'Predictive Model Strategy Gains/Losses'
    plt.legend((a, b), fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=True)
    plt.title('Buy-and-hold Strategy vs. Predictive Model Strategy', fontsize=14)

    return plt.show()


st.set_option('deprecation.showPyplotGlobalUse', False)
if stock_select != 'Select Ticker' and daterange_select != 'Select Range':
    with st.expander(f"The Experiment Graph for {daterange_select}:"): # Experiment Graph
        st.pyplot(model_experiment_plot(df_experiment))
        st.write('What the graph is telling us:')
        st.write(verdict_text)
        # st.write("***")

    with st.expander(f"Confusion matrix:"): # Confusion Matrix
        y_true = y
        y_pred = model.predict(X)
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

        st.pyplot(confusion_plot(X, y, model))
        st.write("Interpreting this matrix:")
        st.markdown(f'There are {cm[0][0]} and {cm[1][1]} ***correctly*** predicted Negative Trends and Positive Trends respectively.')
        st.write(f'There are {cm[0][1]} and {cm[1][0]} ***incorrectly*** predicted Negative Trends and Positive Trends respectively.')

    with st.expander(f"Autocorrelation of {feature_select}:"): # Autocorrelation
        st.pyplot(autocorrelation_chart(stock_select, daterange_select, feature_select))
        st.write("What is autocorrelation and why is it important?\n")
        st.write("Autocorrelation tells us how correlated the feature is with itself in past values. "
                 "The purpose of autocorrelation in a machine learning context is to inform us about whether "
                 "or not we should consider past values in building our model. "
                 "The shaded blue region is the confidence interval and any day that falls into this region is considered statistically insignificant.")
