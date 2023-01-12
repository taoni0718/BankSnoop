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

def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 100, 32
    num_classes = y_train.shape[1]
    maml = MAML(args.input_shape, num_classes)
    model = maml.get_maml_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return model, accuracy

# run an experiment
def run_experiment(repeats=10):
    # load data
    # trainX, trainy, testX, testy = load_dataset()
    max_score = 0
    trainX, trainy, testX, testy = X_train, y_train, X_test, y_test
    # repeat experiment
    scores = list()
    for r in range(repeats):
        model, score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        if score > max_score:
            max_score = score
            model.save_weights("pb_af_closed.h5")
            model.save("pb_af_closed_model.h5")
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    summarize_results(scores)
    
# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

run_experiment()