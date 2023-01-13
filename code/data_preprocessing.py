# cnn model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
import scipy.io
import numpy as np
from sklearn import *
from scipy import stats
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.utils import to_categorical

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

ally = {'raw_y': Y}
df_y = pd.DataFrame(ally)

ord_enc = OrdinalEncoder()
df_y["y_encode"] = ord_enc.fit_transform(df_y[["raw_y"]])
df_y[["raw_y", "y_encode"]].head(5)

Yt = df_y['y_encode'].tolist()
Yt = to_categorical(Yt)
print(Yt.shape)

X_train, X_test, y_train, y_test = model_selection.train_test_split(Xt, Yt, train_size = 0.9, test_size = 0.1, random_state = 100)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

def build_idx_dict(yval):
    idx_dict = {}
    for idx in range(len(yval)):
        class_name = yval[idx]
        if class_name not in idx_dict.keys():
            idx_dict[class_name] = [idx]
        else:
            idx_dict[class_name].append(idx)
            
    return idx_dict

def get_magdata_idx(magdata, idx):
    return magdata[idx]

idx_dict = build_idx_dict(Yu)
idx_dict_keys = list(idx_dict.keys())
magdata_example = get_magdata_idx(Xtu, 5)
print(idx_dict_keys)
print(magdata_example)