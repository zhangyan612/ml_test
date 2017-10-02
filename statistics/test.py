import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

# https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
# data = pd.read_csv('data/AirPassengers.csv')
# print(data.head())
# print('\n Data Types:')
# print(data.dtypes)

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('data/AirPassengers.csv', parse_dates='Month', index_col='Month',date_parser=dateparse)
print(data.head())
