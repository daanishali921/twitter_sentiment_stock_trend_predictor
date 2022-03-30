# Twitter Sentiment Stock Trend Predictor

### Deployed on:

https://twist-informer.herokuapp.com/

### About

As somebody who spends most of his business days day trading, the majority of the thoughts that are behind the day involve wondering what and how market trends will come into fruition and why.
In the past year and a half, the stock market has seen impactful influences from many different avenues; one of them being social media.
Because of this, I built a tool to help day traders like myself gain insight on the influence of twitter trends on stock trends.



### Purpose/Business Objective

Our target user is the day trader who would like to investigate whether or not Twitter trends can serve as a signal to stock market trends.

The user will be able to go onto an interactive website and choose a stock (currently one: **$GME**) which the user wants to learn about. 
The user will then be able to pick a date (currently one date), the machine learning model in the backend will output predictions about stock trends 3 days into the future (which accounts for time lag). 

We use the predictions to develop an experiment; we run 2 strategies:
  1. Buy and hold
  2. Buy and sell given the suggestions of our model
  


### Data Injestion/Wrangling

Using the snscrape Python library, we built a custom function to gather Twitter data organized by date and filtered by an inputted "keyword" (ie. Gamestop). This function also stores and returns the data onto a Pandas DataFrame.

Getting the stock data was very simple, we used a Python library called yfinance and inputted the dates and stock ticker we needed.

The tweet text feature on our DataFrame was rather messy, but we found a general approach to clean up and filter the column of the DataFrame using standard Pandas DataFrame methods as well as further filtering the text using regular expressions to get rid of redundancies. 

We ended up with a DataFrame of over 200,000 rows.
After the processing of features, we aggregated them into a DateTime Indexed DataFrame to be able to fit into the model. We took the average per day of each feature.

### Natural Language Processing/Feature Engineering

In order to meaningfully use tweets as a feature, we need some way to quantify them. Using a prebuilt package called VADER, we were able to perform sentiment analysis on the tweets, giving each tweet a "polarity score" which quantitatively allowed us to determine how positive or negative the tweet was. We used these scores to develop new features for our model. 

#### Interpreting Positive/Negative/Neutral/Compound Score:

Direct information regarding scoring:
https://github.com/cjhutto/vaderSentiment/blob/master/README.rst#about-the-scoring



### Feature List: (Each feature is numerical, none are categorical)
  1. Retweet Count 
  2. Likes Count 
  3. Reply Count 
  4. Positive Tweet Score 
  5. Negative Tweet Score 
  6. Neutral Tweet Score
  7. Compound Tweet Score
  8. Compound Followers (calculated by multiplying average followers and the compound score) 

### Autocorrelation

After cleaning up our DataFrame entirely, it was time for exploring our data/features. By the nature of a time-series dataset, it was essential to consider the time dependent nature of each of our features; we figured out the Autocorrelation of each feature. Some features had an autocorrelation that was statistically significant and others did not. This all played a part in considering the time lag for our model.

We can see the autocorrelation chart(s) for each feature as well as a simple explanation to why autocorrelation might be important on the webapp. 

### Modeling

Before modeling, we merge the stock data with the Twitter data into one DataFrame. 


Our ultimate goal is to use the features we've built with Twitter data to correctly predict a positive or negative stock trend. This calls for a Binary Classifier. 


Luckily, our data was more or less balanced which means we did not have to perform any over/undersampling (or other techniques to deal with imbalanced data).


We trained a Random Forest Classifier whose pipeline consisted of only the Standard Scaler so we don't throw off our model. 


#### Cross-Validation:
Since our dataset is time sensitive, **we could not use ordinary train/test split or standard cross validation methods; we used cross validation on a rolling basis**. We start with a small subset of data for training purpose, forecast for the later data points and then checking the accuracy for the forecasted data points. The same forecasted data points are then eincluded as a part of the next training dataset and subsequent data points are forecasted. 

#### Precision, Recall, Accuracy?
Naturally, dealing with a binary classifier means there is a choice that needs to be made in regards to which metric is most important: 
  1. Accuracy
  2. Recall
  3. Precision

Each have their perks. The question being what is the most important thing to minimize, False Positives or False Negatives? In the world of the stock market, one has the ability to make money on both a negative trend (by shorting a stock before the trend occurs) as well as a positive trend (by buying low before the trend occurs). This made picking the most important metric a tricky decision, but ultimately picked accuracy as the choice due to the doublesided nature of our problem. 

#### Visualizations:
After the model was built and tested, we built a confusion matrix giving us visual information about the performance of our model:

![image](https://user-images.githubusercontent.com/60590897/160923616-968a9fb1-3d82-41f7-ab19-cbb81c4b683d.png)


Even though our model was slightler better than a baseline model, the experiment we built that guided the buying and selling of the stock turned out to be underperforming the simple buy/hold strategy:
**INFORMATION ABOUT GRAPH INTERPRETATION ON THE WEBAPP**

![image](https://user-images.githubusercontent.com/60590897/160923899-a247e3b0-39fe-4190-9fd5-b0f938638a69.png)

_INFORMATION ABOUT GRAPH INTERPRETATION ON THE WEBAPP_


### Concluding Thoughts
Even though the model does provide _some_ signal within Twitter/tweets towards stock trends, the nature of the stock market is much too chaotic to predict with any confident levels of accuracy.
