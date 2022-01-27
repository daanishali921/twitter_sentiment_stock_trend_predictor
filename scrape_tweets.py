import snscrape.modules.twitter as sntwitter
import os
import pandas as pd
from datetime import date


def scrape_tweets(start_date, end_date, keyword, current_dir, tweet_limit=1, iteration=0):
    if not os.path.exists(current_dir):  # Creates directory in current directory if doesn't already exist
        os.mkdir(current_dir)

    file_path = os.path.join(current_dir, f'keyword:{keyword}__start:{start_date}_end:{end_date}__iter:{iteration}.csv')

    tweet_list = []
    for i, tweet in enumerate(
            sntwitter.TwitterSearchScraper(f'{keyword} since:{start_date} until:{end_date}').get_items()):
        if i > tweet_limit:
            break
        tweet_list.append([tweet.date,  # Appending all tweet data into a list of list
                           tweet.id,
                           tweet.content,
                           tweet.user.username,
                           tweet.user.followersCount,
                           tweet.hashtags,
                           tweet.cashtags,
                           tweet.lang,
                           tweet.retweetCount,
                           tweet.likeCount,
                           tweet.replyCount])

    df_tweets = pd.DataFrame(tweet_list, columns=['Datetime',  # Creating df of tweet data
                                                  'Tweet Id',
                                                  'Text',
                                                  'Username',
                                                  'Followers Count',
                                                  'Hashtags',
                                                  'Cashtags',
                                                  'Language',
                                                  'Retweet Count',
                                                  'Likes Count',
                                                  'Reply Count'])

    for i in range(1001):
        if not os.path.isfile(file_path):
            df_tweets.to_csv(file_path, index=False)
            break
        else:
            file_path = os.path.join(current_dir,
                                     f'keyword:{keyword}__start:{start_date}_end:{end_date}__limit:{tweet_limit}__iter:{i + 1}.csv')

    if os.path.isfile(file_path):
        return print(f'Successfully saved DataFrame to {file_path}')
    return print('DataFrame not saved -- possible error has occurred.')


start_time_tsla = date(2020, 1, 1).strftime('%Y-%m-%d')
end_time_tsla = date(2022, 1, 1).strftime('%Y-%m-%d')

if __name__ == '__main__':
    main()
