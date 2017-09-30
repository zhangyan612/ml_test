from __future__ import print_function
import numpy as np
import pandas as pd
# import random, timeit
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
from talib.abstract import *
from sklearn.externals import joblib
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.layers.recurrent import LSTM
# from keras.optimizers import RMSprop, Adam
import quandl


# Download data from online source
def read_convert_data():
    prices = quandl.get("ECB/EURUSD")
    prices.to_pickle('data/EURUSD_1day.pkl')
    return


def load_data():
    np.random.seed(1335)  # for reproducibility
    np.set_printoptions(precision=5, suppress=True, linewidth=150)

    price = np.sin(np.arange(300) / 30.0)  # sine prices
    return price


def load_stock_data():
    prices = pd.read_pickle('data/EURUSD_1day.pkl')
    print('total data set: %s' % len(prices.index))
    x_train = prices.iloc[-650:-200, ]
    print('training data set: %s' % len(x_train.index))
    print('start with %s' % x_train.index[0])
    return x_train

def draw_data_graph():
    x = load_stock_data()
    plt.plot(x)
    plt.show()


def init_state(data):
    close = data
    diff = np.diff(data)
    diff = np.insert(diff, 0, 0)

    # --- Preprocess data
    xdata = np.column_stack((close, diff))
    xdata = np.nan_to_num(xdata)
    scaler = preprocessing.StandardScaler()
    xdata = scaler.fit_transform(xdata)
    state = xdata[0:1, :]
    return state, xdata

def init_state_stock(indata):
    close = indata['Value'].values
    diff = np.diff(close)
    diff = np.insert(diff, 0, 0)

    # --- Preprocess data
    xdata = np.column_stack((close, diff))
    xdata = np.nan_to_num(xdata)
    scaler = preprocessing.StandardScaler()
    # xdata = np.expand_dims(scaler.fit_transform(xdata), axis=1)
    # joblib.dump(scaler, 'data/scaler.pkl')
    # state = xdata[0:1, 0:1, :]
    xdata = scaler.fit_transform(xdata)
    state = xdata[0:1, :]
    return state, xdata



if __name__ == "__main__":
    # read_convert_data(symbol='EURUSD_1day')  # run once to read indata, resample and convert to pickle
    # random_data = load_data()
    # state, xdata = init_state(random_data)
    # print(state, xdata)

    # test_data = load_stock_data()
    # state, xdata = init_state_stock(test_data)
    # print(state, xdata)
    # state, xdata = init_state(test_data)
    import random

    print(random.random())
    # print(state, xdata)
    # read_convert_data()
    # draw_data_graph()
