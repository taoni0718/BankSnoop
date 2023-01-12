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

dataset = scipy.io.loadmat('../input/power-bank-app-fingerprinting/app_fingerprinting_ego.mat')
#dataset = scipy.io.loadmat('App_dataset_game_2_0620.mat')
X = dataset['Farr_all']
Yarr = dataset['y_all']

#print("X: ", X)
#print("Y: ", Yarr)

Y = []
for i in range(len(Yarr)):
    Y.append(Yarr[i][0][0])

print(X.shape)

Xt = []
for i in range(X.shape[0]):
    tmp = np.hstack((X[i][0], X[i][1], X[i][2]))
    Xt.append(tmp)
    
Xt = np.array(Xt)
print(Xt.shape)