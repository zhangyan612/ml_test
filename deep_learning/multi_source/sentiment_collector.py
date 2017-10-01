import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata


# this will collect news data on a stock and combine them into time series sentiment features

# Reading the saved data pickle file
df_stocks = pd.read_pickle('Data/Pickled_ten_year_filtered_data.pkl')
df_stocks['prices'] = df_stocks['adj close'].apply(np.int64)

# selecting the prices and articles, combine
df_stocks = df_stocks[['prices', 'articles']]
df_stocks['articles'] = df_stocks['articles'].map(lambda x: x.lstrip('.-'))

# print(df_stocks)


# Adding new columns to the data frame
df = df_stocks[['prices']].copy()
df["compound"] = ''
df["neg"] = ''
df["neu"] = ''
df["pos"] = ''

# use sentiment analyzer
sid = SentimentIntensityAnalyzer()
for date, row in df_stocks.T.iteritems():
    try:
        sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles'])#.encode('ascii','ignore')
        ss = sid.polarity_scores(sentence)
        df.set_value(date, 'compound', ss['compound'])
        df.set_value(date, 'neg', ss['neg'])
        df.set_value(date, 'neu', ss['neu'])
        df.set_value(date, 'pos', ss['pos'])
    except TypeError:
        print(df_stocks.loc[date, 'articles'])
        print(date)

print(df)
df.to_pickle('sentiment.pkl')
