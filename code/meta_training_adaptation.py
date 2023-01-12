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

inner_optimizer = optimizers.Adam(args.inner_lr)
outer_optimizer = optimizers.Adam(args.outer_lr)

maml = MAML(args.input_shape, args.n_way)
# 验证次数可以少一些，不需要每次都更新这么多
val_data.steps = 10

for e in range(args.epochs):

    train_progbar = utils.Progbar(train_data.steps)
    val_progbar = utils.Progbar(val_data.steps)
    print('\nEpoch {}/{}'.format(e+1, args.epochs))

    train_meta_loss = []
    train_meta_acc = []
    val_meta_loss = []
    val_meta_acc = []

    for i in range(train_data.steps):
        batch_train_loss, acc = maml.train_on_batch_mag(train_data.get_one_batch_mag(),
                                                    inner_optimizer,
                                                    inner_step=1,
                                                    outer_optimizer=outer_optimizer)

        train_meta_loss.append(batch_train_loss)
        train_meta_acc.append(acc)
        train_progbar.update(i+1, [('loss', np.mean(train_meta_loss)),
                                   ('accuracy', np.mean(train_meta_acc))])

    for i in range(val_data.steps):
        batch_val_loss, val_acc = maml.train_on_batch_mag(val_data.get_one_batch_mag(), inner_optimizer, inner_step=3)

        val_meta_loss.append(batch_val_loss)
        val_meta_acc.append(val_acc)
        val_progbar.update(i+1, [('val_loss', np.mean(val_meta_loss)),
                                 ('val_accuracy', np.mean(val_meta_acc))])

    maml.meta_model.save_weights("pb_af_maml.h5")


from tensorflow.keras import optimizers, utils, metrics
import tensorflow as tf
import numpy as np

import os

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

val_data = MAMLDataLoaderMag(idx_dict_keys, args.val_batch_size, args.n_way, args.k_shot, args.q_query)
#print(val_data.file_list)

mnist_model = MAML(args.input_shape, args.n_way)
maml = MAML(args.input_shape, args.n_way)

# 对比测试
trc_acc = []
few_acc = []
for i in range(5):
    # mnist weights
    mnist_model.meta_model.load_weights("./pb_af_closed.h5")
    optimizer = optimizers.Adam(args.inner_lr)
    val_loss, val_acc = mnist_model.train_on_batch_mag(val_data.get_one_batch_mag(), inner_optimizer=optimizer, inner_step=3)
    print("Model with mnist initialize weight train for 3 step, val loss: {:.4f}, accuracy: {:.4f}.".format(val_loss, val_acc))

    mnist_model.meta_model.load_weights("./pb_af_closed.h5")
    optimizer = optimizers.Adam(args.inner_lr)
    val_loss, val_acc = mnist_model.train_on_batch_mag(val_data.get_one_batch_mag(), inner_optimizer=optimizer, inner_step=10)
    trc_acc.append(val_acc)
    print("Model with mnist initialize weight train for 5 step, val loss: {:.4f}, accuracy: {:.4f}.".format(val_loss, val_acc))

    # maml weights
    maml.meta_model.load_weights("./pb_af_maml.h5")
    optimizer = optimizers.Adam(args.inner_lr)
    val_loss, val_acc = maml.train_on_batch_mag(val_data.get_one_batch_mag(), inner_optimizer=optimizer, inner_step=3)
    print("Model with maml weight train for 3 step, val loss: {:.4f}, accuracy: {:.4f}.".format(val_loss, val_acc))

    maml.meta_model.load_weights("./pb_af_maml.h5")
    optimizer = optimizers.Adam(args.inner_lr)
    val_loss, val_acc = maml.train_on_batch_mag(val_data.get_one_batch_mag(), inner_optimizer=optimizer, inner_step=10)
    few_acc.append(val_acc)
    print("Model with maml weight train for 5 step, val loss: {:.4f}, accuracy: {:.4f}.".format(val_loss, val_acc))
    
print(np.mean(trc_acc))
print(np.std(trc_acc))
print(np.mean(few_acc))
print(np.std(few_acc))