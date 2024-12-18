import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # index of GPU machine (0 or 1)
import tensorflow as tf
print(tf.test.gpu_device_name())

import numpy as np
import h5py

from sklearn.preprocessing import binarize

# ---------------------------------------------------------------------------------------------------------------------------------------
# read test data
test_set_ = h5py.File('./../../spectra_Roland/samples/test_samples/real_samples/SNRfull_WLfull_WL_fixed_range_fixed_test_ygap.hdf5', 'r')
# 'EW_r_2796', 'EW_r_2803', 'EW_r_total', 'SNR', 'absorber_true', 'cent_WL_2796', 'cent_pix_2796', 'flux', 'master_wave', 'redshift'

X_test = test_set_['flux'][:]  # spectra
y_test_class = test_set_['absorber_true'][:]  # classification labels
y_test_reg = test_set_['cent_WL_2796'][:]  # regression labels
# print(X_test[0])
# print(y_test_class[0])
# print(y_test_reg[0])

# ---------------------------------------------------------------------------------------------------------------------------------------
# read trained model
model_path = './checkpoints/first_attempt_Rolspec_5_epochs_526_batchsize/'
model = tf.keras.models.load_model(model_path)

# getting predictions
print('Predicting... ')
y_pred = model.predict(X_test)
print(' ')

# classification
y_pred_class_prob = y_pred[0]
class_label_threshhold = 0.5
y_pred_class = binarize(y_pred_class_prob, threshold=class_label_threshhold)
print('y_pred_class_prob.shape ', y_pred_class_prob.shape)
print('y_pred_class.shape ', y_pred_class.shape)
print('np.unique(y_pred_class) ', np.unique(y_pred_class))

# regression
y_pred_reg = y_pred[1]
# print('y_pred_reg.shape before ', y_pred_reg.shape)
# y_pred_reg = y_pred_reg.reshape(y_test_reg.shape)
# print('y_pred_reg.shape after ', y_pred_reg.shape)

# saving predictions
np.save(model_path + 'y_pred_class_prob.npy', y_pred_class_prob, allow_pickle=True)
np.save(model_path + 'y_pred_class.npy', y_pred_class, allow_pickle=True)
np.save(model_path + 'y_pred_reg.npy', y_pred_reg, allow_pickle=True)

# --------------------------------------------------------------------------------------------------------------------------------