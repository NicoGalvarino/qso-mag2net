# import pandas as pd
import numpy as np
import h5py
# from tensorflow.keras.models import load_model
# from keras.layers import TFSMLayer
# import tensorflow as tf

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
import seaborn as sns

from generators import generator_fiducial_modelx

# read test data
test_set = h5py.File('./../../spectra_Roland/samples/test_samples/real_samples/SNRfull_WLfull_WL_fixed_range_fixed_test_ygap.hdf5', 'r')

# model = tf.saved_model.load('./checkpoints/first_attempt_Rolspec_5_epochs_526_batchsize/')
# print(type(model))

absorber_true = test_set['absorber_true'][:]
mask = absorber_true == 1
# sample = {'X_test': test_set['X_test'][mask],
#           'y_test': test_set['y_test'][mask]
# }
sample_size = len(test_set['flux'])
# test_set = generator_fiducial_model.DataGenerator(range(0, sample_size), sample, 'absorber_true', 'cent_WL_2796')

# X_test = sample['X_test'][:]
# y_test = sample['y_test'][:]

# y_pred = model.predict(X_test)

# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Test Loss: {loss}')
# print(f'Test Accuracy: {accuracy}')

# print(' ')
# print(' ')
# print(' ')
# print(test_set.keys())
# label_abs = test_set['absorber_true']
# print(label_abs.name)
# print(label_abs.shape)
# print(label_abs.dtype)
# [print(label_abs[i]) for i in range(10)]
# print(np.unique(label_abs))