from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# 1. Polarity Score Categories as a column on DataFrame
# 2. Creating categorical polarity column

def polarityCategories_textblob(tup):
    if tup[0] < -0.05:
        return 'Negative'
    elif tup[0] > 0.05:
        return 'Positive'
    else:
        return 'Neutral'

# using Textblob NLP module
def polarityColumns_textblob(df_tweets):
    df_tweets['Polarity/Subjectivity Scores'] = df_tweets['Text'].apply(lambda text: TextBlob(text).sentiment)
    df_tweets_filtered = df_tweets[df_tweets['Polarity/Subjectivity Scores'] != (0.0, 0.0)]
    df_tweets_filtered = df_tweets_filtered.copy()
    df_tweets_filtered['Polarity Categories'] = df_tweets_filtered['Polarity/Subjectivity Scores'].apply(
        lambda x: polarityCategories(x))

    print(df_tweets_filtered.value_counts('Polarity Categories'))

    return df_tweets_filtered


def polarityColumns_vader(df_tweets):
    df_tweets['Sentiment'] = df_tweets['Text'].apply(lambda text: SentimentIntensityAnalyzer().polarity_scores(text))
    #     df_tweets_filtered = df_tweets_filtered.copy()
    #     df_tweets_filtered['Polarity Categories'] = df_tweets_filtered['Polarity/Subjectivity Scores'].apply(lambda x: polarityCategories_vader(x))

    #     print(df_tweets_filtered.value_counts('Polarity Categories'))

    return df_tweets


def polarityCategories_vader(tup):
    if tup[0] < -0.05:
        return 'Negative'
    elif tup[0] > 0.05:
        return 'Positive'
    else:
        return 'Neutral'