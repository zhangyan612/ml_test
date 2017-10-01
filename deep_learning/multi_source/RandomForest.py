import numpy as np
import pandas as pd
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import unicodedata
# from nltk.classify import NaiveBayesClassifier
# from nltk.corpus import subjectivity
# from nltk.sentiment import SentimentAnalyzer
# from nltk.sentiment.util import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix
from treeinterpreter import treeinterpreter as ti
import matplotlib.pyplot as plt

# read pickle file
df = pd.read_pickle('sentiment.pkl')
print(df)


train_start_date = '2007-01-01'
train_end_date = '2014-12-31'
test_start_date = '2015-01-01'
test_end_date = '2016-12-31'
train = df.ix[train_start_date : train_end_date]
test = df.ix[test_start_date:test_end_date]

# initialize sentiment data
sentiment_score_list = []
for date, row in train.T.iteritems():
    #sentiment_score = np.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
    sentiment_score = np.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
    sentiment_score_list.append(sentiment_score)
numpy_df_train = np.asarray(sentiment_score_list)


sentiment_score_list2 = []
for date, row in test.T.iteritems():
    #sentiment_score = np.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
    sentiment_score = np.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
    sentiment_score_list2.append(sentiment_score)
numpy_df_test = np.asarray(sentiment_score_list2)

# prepare train and test data
y_train = pd.DataFrame(train['prices'])
y_test = pd.DataFrame(test['prices'])


# random forest
rf = RandomForestRegressor()
rf.fit(numpy_df_train, y_train)

print(rf.feature_importances_)

prediction, bias, contributions = ti.predict(rf, numpy_df_test)
# print(prediction)
# print(contributions)

# made prediction into df
idx = pd.date_range(test_start_date, test_end_date)
predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['prices'])

predictions_plot = predictions_df.plot()

# fig = y_test.plot(ax = predictions_plot).get_figure()
# fig.savefig("graphs/random forest without smoothing.png")

ax = predictions_df.rename(columns={"prices": "predicted_price"}).plot(title='Random Forest predicted prices 8-2 years')
ax.set_xlabel("Dates")
ax.set_ylabel("Stock Prices")
fig = y_test.rename(columns={"prices": "actual_price"}).plot(ax = ax).get_figure()
fig.savefig("graphs/random forest without smoothing.png")

