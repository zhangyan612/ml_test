import numpy as np
import pandas as pd
import pickle


# Write df to pickle

# df = pd.DataFrame(np.random.rand(5, 5))
# print(df)
# df.to_pickle('test.pkl')


# read pickle file
df = pd.read_pickle('test.pkl')
print(df)