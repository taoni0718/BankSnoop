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

unmonitored_dataset = scipy.io.loadmat('../input/power-bank-app-fingerprinting/app_fingerprinting_belkin.mat')
#dataset = scipy.io.loadmat('App_dataset_game_2_0620.mat')
Xu = unmonitored_dataset['Farr_all']
Yarru = unmonitored_dataset['y_all']

#print("X: ", X)
#print("Y: ", Yarr)

Yu = []
for i in range(len(Yarru)):
    Yu.append(Yarru[i][0][0])

print(Xu.shape)
print(len(Yu))

Xtu = []
for i in range(Xu.shape[0]):
    tmp = np.hstack((Xu[i][0], Xu[i][1], Xu[i][2]))
    Xtu.append(tmp)
    
Xtu = np.array(Xtu)
print(Xtu.shape)

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

allyu = {'raw_y': Yu}
df_yu = pd.DataFrame(allyu)

ord_encu = OrdinalEncoder()
df_yu["y_encode"] = ord_encu.fit_transform(df_yu[["raw_y"]])
df_yu[["raw_y", "y_encode"]].head(5)

Ytu = df_yu['y_encode'].tolist()
Ytu = to_categorical(Ytu)
print(Ytu.shape)