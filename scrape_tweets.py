import snscrape.modules.twitter as sntwitter
import os
import pandas as pd
from datetime import date



def scrapeTweets(start, stop, keyword, directory, tweet_limit=1):
    if not os.path.exists(directory):  # Creates directory in current directory if doesn't already exist
        os.mkdir(directory)

    file_path = os.path.join(directory, f'keyword:{keyword}__start:{start}_end:{stop}__limit:{tweet_limit}.csv')

    tweet_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{keyword} since:{start} until:{stop}').get_items()):
        if i > tweet_limit:
            break
        tweet_list.append([tweet.date,  # Appending all tweet data into a list of list
                           tweet.id,
                           tweet.content,
                           tweet.user.username,
                           tweet.user.followersCount,
                           tweet.hashtags,
                           tweet.cashtags,
                           tweet.lang])

    df_tweets = pd.DataFrame(tweet_list, columns=['Datetime',  # Creating df of tweet data
                                                  'Tweet Id',
                                                  'Text',
                                                  'Username',
                                                  'Followers Count',
                                                  'Hashtags',
                                                  'Cashtags',
                                                  'Language'])

    df_tweets.to_csv(file_path, index=False)  # Writing df_tweets into new csv file

    if os.path.isfile(file_path) == True:
        return print(f'Successfully saved DataFrame to {file_path}')
    else:
        return print('DataFrame not saved -- possible error has occurred.')

start_time_tsla = date(2020, 1, 1).strftime('%Y-%m-%d')
end_time_tsla = date(2022, 1, 1).strftime('%Y-%m-%d')

if __name__ == '__main__':
    scrapeTweets(start=start_time_tsla, stop=end_time_tsla, keyword='TSLA', directory='tweets_TSLA', tweet_limit=50000)
