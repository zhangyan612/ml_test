from sklearn.neural_network import MLPClassifier
from datetime import datetime, timedelta
from treeinterpreter import treeinterpreter as ti
import numpy as np
import pandas as pd
from deep_learning.multi_source.offset_value import *

# read pickle file
df = pd.read_pickle('sentiment.pkl')
print(df)

# average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'prices']
# # Converting string to date time
# temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
# # Adding one day from date time
# difference = temp_date + timedelta(days=1)
# # Converting again date time to string
# temp_date = difference.strftime('%Y-%m-%d')

# start_year = datetime.strptime(train_start_date, "%Y-%m-%d").date().month

years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
prediction_list = []
for year in years:
    # Splitting the training and testing data
    train_start_date = str(year) + '-01-01'
    train_end_date = str(year) + '-10-31'
    test_start_date = str(year) + '-11-01'
    test_end_date = str(year) + '-12-31'
    train = df.ix[train_start_date: train_end_date]
    test = df.ix[test_start_date:test_end_date]

    # Calculating the sentiment score
    sentiment_score_list = []
    for date, row in train.T.iteritems():
        sentiment_score = np.asarray(
            [df.loc[date, 'compound'], df.loc[date, 'neg'], df.loc[date, 'neu'], df.loc[date, 'pos']])
        # sentiment_score = np.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
        sentiment_score_list.append(sentiment_score)
    numpy_df_train = np.asarray(sentiment_score_list)
    sentiment_score_list = []
    for date, row in test.T.iteritems():
        sentiment_score = np.asarray(
            [df.loc[date, 'compound'], df.loc[date, 'neg'], df.loc[date, 'neu'], df.loc[date, 'pos']])
        # sentiment_score = np.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
        sentiment_score_list.append(sentiment_score)
    numpy_df_test = np.asarray(sentiment_score_list)

    # Generating models
    mlpc = MLPClassifier(hidden_layer_sizes=(100, 200, 100), activation='relu',
                         solver='lbfgs', alpha=0.005, learning_rate_init=0.001, shuffle=False)  # span = 20 # best 1
    mlpc.fit(numpy_df_train, train['prices'])
    prediction = mlpc.predict(numpy_df_test)

    prediction_list.append(prediction)
    # print train_start_date + ' ' + train_end_date + ' ' + test_start_date + ' ' + test_end_date
    idx = pd.date_range(test_start_date, test_end_date)
    # print year
    predictions_df_list = pd.DataFrame(data=prediction[0:], index=idx, columns=['prices'])

    difference_test_predicted_prices = offset_value(test_start_date, test, predictions_df_list)
    # Adding offset to all the advpredictions_df price values
    predictions_df_list['prices'] = predictions_df_list['prices'] + difference_test_predicted_prices
    print(predictions_df_list)

    # Smoothing the plot
    predictions_df_list['ewma'] = pd.ewma(predictions_df_list["prices"], span=20, freq="D")
    predictions_df_list['actual_value'] = test['prices']
    predictions_df_list['actual_value_ewma'] = pd.ewma(predictions_df_list["actual_value"], span=20, freq="D")
    # Changing column names
    predictions_df_list.columns = ['predicted_price', 'average_predicted_price', 'actual_price', 'average_actual_price']
    ax = predictions_df_list.plot()
    fig = ax.get_figure()
    fig.savefig('graphs/' + str(year) + '_MLP_predicted.png')

    predictions_df_list_average = predictions_df_list[['average_predicted_price', 'average_actual_price']]
    ax = predictions_df_list_average.plot()
    fig = ax.get_figure()
    fig.savefig('graphs/' + str(year) + '_MLP_average_predicted.png')



# mlpc = MLPClassifier(hidden_layer_sizes=(100, 200, 100), activation='tanh',
#                          solver='lbfgs', alpha=0.010, learning_rate_init = 0.001, shuffle=False)
# mlpc = MLPClassifier(hidden_layer_sizes=(100, 200, 100), activation='relu',
#                          solver='lbfgs', alpha=0.010, learning_rate_init = 0.001, shuffle=False) # span = 20
# mlpc = MLPClassifier(hidden_layer_sizes=(100, 200, 100), activation='relu',
#                          solver='lbfgs', alpha=0.005, learning_rate_init = 0.001, shuffle=False) # span = 20 # best 1
# mlpc = MLPClassifier(hidden_layer_sizes=(100, 200, 50), activation='relu',
#                          solver='lbfgs', alpha=0.005, learning_rate_init = 0.001, shuffle=False)
#
# # checking the performance of training data itself
# prediction, bias, contributions = ti.predict(rf, numpy_df_train)
# idx = pd.date_range(train_start_date, train_end_date)
# predictions_df1 = pd.DataFrame(data=prediction[0:], index = idx, columns=['prices'])
# predictions_df1.plot()
# train['prices'].plot()