import pandas as pd
import re


# Creating function to clean dataframes programatically

def data_wrangle(dataframe_list):
    df_concat = pd.concat(dataframe_list)  # Concatenate all the DataFrames from the list of DataFrames
    df_filter = df_concat[df_concat['Language'] == 'en'][['Datetime',  # Filter via Language = 'en'
                                                          'Tweet Id',  # Remove unwanted columns
                                                          'Text',
                                                          'Username',
                                                          'Followers Count']]
    df_clean = df_filter.astype({'Tweet Id': str}) \
        .dropna() \
        .drop_duplicates() \
        .reset_index(drop=True)

    df_clean['Text'] = (df_clean['Text']  # Cleans out redundant string characters within each tweet
                        .apply(lambda x: ' '.join(re.sub(r'https\S+', '', x)
                                                  .replace('\n', ' ')
                                                  .split()
                                                  )
                               )
                        )
    return df_clean


def data_wrangle_single(dataframe):
    df_filter = dataframe[dataframe['Language'] == 'en'][['Datetime',  # Filter via Language = 'en'
                                                          'Tweet Id',  # Remove unwanted columns
                                                          'Text',
                                                          'Username',
                                                          'Followers Count',
                                                          'Retweet Count',
                                                          'Likes Count',
                                                          'Reply Count']]
    df_clean = df_filter.astype({'Tweet Id': str}) \
        .dropna() \
        .drop_duplicates() \
        .reset_index(drop=True)

    df_clean['Text'] = (df_clean['Text']  # Cleans out redundant string characters within each tweet
                        .apply(lambda x: ' '.join(re.sub(r'https\S+', '', x)
                                                  .replace('\n', ' ')
                                                  .split()
                                                  )
                               )
                        )

    df_clean['Datetime'] = pd.to_datetime(df_clean['Datetime']).dt.floor('d').dt.tz_localize(tz=None)

    return df_clean.set_index('Datetime').sort_index()


def data_wrangle_list(dataframe_list):
    df_concat = pd.concat(dataframe_list)  # Concatenate all the DataFrames from the list of DataFrames
    df_filter = df_concat[df_concat['Language'] == 'en'][['Datetime',  # Filter via Language = 'en'
                                                          'Tweet Id',  # Remove unwanted columns
                                                          'Text',
                                                          'Username',
                                                          'Followers Count',
                                                          'Retweet Count',
                                                          'Likes Count',
                                                          'Reply Count']]
    df_clean = df_filter.astype({'Tweet Id': str}) \
        .dropna() \
        .drop_duplicates() \
        .reset_index(drop=True)

    df_clean['Text'] = (df_clean['Text']  # Cleans out redundant string characters within each tweet
                        .apply(lambda x: ' '.join(re.sub(r'https\S+', '', x)
                                                  .replace('\n', ' ')
                                                  .split()
                                                  )
                               )
                        )

    df_clean['Datetime'] = pd.to_datetime(df_clean['Datetime']).dt.floor('d').dt.tz_localize(tz=None)

    return df_clean.set_index('Datetime').sort_index()
