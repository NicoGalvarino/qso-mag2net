import tensorflow as tf
import numpy as np
import h5py
import pandas as pd

from sklearn.preprocessing import binarize
from sklearn.metrics import (classification_report, accuracy_score, precision_score, recall_score, f1_score, 
                            mean_absolute_error, confusion_matrix, roc_curve, auc)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
import mpl_scatter_density
import seaborn as sns
import os

# ---------------------------------------------------------------------------------------------------------------------------------------
print('Reading test data \n')
test_set_ = h5py.File('./../../../spectra_Roland/samples/test_samples/real_samples/SNRfull_WLfull_WL_fixed_range_fixed_test_ygap.hdf5', 'r')
# 'EW_r_2796', 'EW_r_2803', 'EW_r_total', 'SNR', 'absorber_true', 'cent_WL_2796', 'cent_pix_2796', 'flux', 'master_wave', 'redshift'

X_test = test_set_['flux'][:]  # spectra
y_test_class = test_set_['absorber_true'][:]  # classification labels
y_test_reg = test_set_['cent_WL_2796'][:]  # regression labels

# ---------------------------------------------------------------------------------------------------------------------------------------

checkpoint_files = [os.path.splitext(f)[0] for f in os.listdir('./checkpoints') if os.path.isfile(os.path.join('./checkpoints', f))]
print('Trained models:', checkpoint_files)

model_name = input('Which model to evaluate? ')
model_path = './checkpoints/' + model_name + '.keras'
model = tf.keras.models.load_model(model_path)

# ---------------------------------------------------------------------------------------------------------------------------------------

prediction_folders = [f for f in os.listdir('./predictions') if os.path.isdir(os.path.join('./predictions', f))]

# predictions don't exist yet
if model_name not in prediction_folders:

	os.makedirs(os.path.join('./predictions', model_name))
	
	print('Predicting... \n')
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

	print('Saving predictions \n')

	pred_path = './predictions/' + model_name + '/'
	np.save(pred_path + 'y_pred_class_prob.npy', y_pred_class_prob, allow_pickle=True)
	np.save(pred_path + 'y_pred_class.npy', y_pred_class, allow_pickle=True)
	np.save(pred_path + 'y_pred_reg.npy', y_pred_reg, allow_pickle=True)

# predictions exist
else:
	print('Reading predictions \n')
	pred_path = './predictions/' + model_name + '/'
	y_pred_class_prob = np.load(pred_path + 'y_pred_class_prob.npy')
	y_pred_class = np.load(pred_path + 'y_pred_class.npy')
	y_pred_reg = np.load(pred_path + 'y_pred_reg.npy')

# ---------------------------------------------------------------------------------------------------------------------------------------
# classification evaluation
print('Classification \n')

# loss, accuracy = model.evaluate(X_test, y_test_class)
# print(f'Test Loss: {loss}')
# print(f'Test Accuracy: {accuracy}')

accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)
f1_score = f1_score(y_test_class, y_pred_class)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1_score}')
print(classification_report(y_test_class, y_pred_class))

# -----------------------------------------------------------------------------------
# regression evaluation
print('\n Regression \n')

mae = mean_absolute_error(y_test_reg, y_pred_reg)
print(f'MAE: {mae}')

# -----------------------------------------------------------------------------------
plots_folders = [f for f in os.listdir('./model_results_on_test_set') if os.path.isdir(os.path.join('./model_results_on_test_set', f))]
if model_name not in plots_folders:
	os.makedirs(os.path.join('./model_results_on_test_set', model_name))

plot_path = './model_results_on_test_set/' + model_name + '/'

with open(plot_path + 'evaluation_scores.txt', 'w') as f:
	f.write(f'Classification Evaluation:\n')
	f.write(f'Accuracy: {accuracy}\n')
	f.write(f'Precision: {precision}\n')
	f.write(f'Recall: {recall}\n')
	f.write(f'F1-score: {f1_score}\n\n')
	f.write(f'Regression Evaluation:\n')
	f.write(f'MAE: {mae}\n')

# ROC curve ----------

fig_c, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 4.5))

fpr, tpr, thresholds = roc_curve(y_test_class, y_pred_class_prob)

ax1.plot(fpr, tpr, label = f'Area under the curve: {auc(fpr, tpr):.3f}', 
        #  ("Area_under the curve :", np.round(auc(fpr, tpr), 3)), 
         color="r", linewidth=2)
ax1.plot([1, 0], [1, 0], linestyle="dashed", color="k")

ax1.legend(loc="best", fontsize=13)

ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title("ROC - Curve & Area Under the Curve", fontsize=20)

# Confusion matrix ----------

sns.heatmap(confusion_matrix(y_test_class, y_pred_class, normalize='true'), 
            annot=True, cmap="Blues", linecolor="k", linewidths=3, ax=ax2)

ax2.set_title("Confusion Matrix", fontsize=20)
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')

fig_c.tight_layout()
fig_c.savefig(plot_path + 'results_classification.png', dpi=400)

# plot regression ----------------------------------------------------------------
# lambda true vs pred

fig_r = plt.figure(figsize=(10, 5))
ax = fig_r.add_subplot(1, 1, 1)

white_plasma = LinearSegmentedColormap.from_list('white_plasma', 
                                                 ['white'] + plt.cm.plasma(np.linspace(0, 1, 256)).tolist())

nonnan_y_pred = np.where(~np.isnan(y_test_reg))[0]
nonnan_y_test = np.where(~np.isnan(y_pred_reg))[0]
nonnan_idx = np.intersect1d(nonnan_y_pred, nonnan_y_test)
y_test_reg_nonnan = y_test_reg[nonnan_idx]
y_pred_reg_nonnan = y_test_reg[nonnan_idx]
density_range = np.histogram2d(y_test_reg_nonnan, y_pred_reg_nonnan)[0]

vmin = max(1, density_range[density_range > 0].min())
vmax = density_range.max()
print(vmin, vmax)

density = ax.scatter(y_test_reg_nonnan, y_pred_reg_nonnan, 
					#  c=np.log10(np.histogram2d(y_test_reg_nonnan, y_pred_reg_nonnan, bins=50)[0].flatten()), 
					 cmap=white_plasma, norm=LogNorm(vmin=vmin, vmax=vmax))

fig_r.colorbar(density, label='Count (log scale)')
ax.plot([y_test_reg_nonnan.min(), y_test_reg_nonnan.max()], 
        [y_pred_reg_nonnan.min(), y_pred_reg_nonnan.max()], 
        lw=1, c='k', alpha=0.4)

ax.set_xlabel(r'True $\lambda$')
ax.set_ylabel(r'Predicted $\lambda$')
ax.set_title(r'True vs. Predicted $\lambda$')

fig_r.tight_layout()
fig_r.savefig(plot_path + 'results_regression.png', dpi=400)


# ----------------------------------------------------------------
# plot training
logs = pd.read_csv('./logfiles/' + model_name + '_history.csv')

fig_c, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15, 5))

ax1.grid(True)
ax2.grid(True)

ax1.set_title('Classification')
ax1.plot(logs['epoch']+1, logs['out_class_loss'], c='darkblue', label='Training loss')
ax1.plot(logs['epoch']+1, logs['val_out_class_loss'], c='orange', label='Validation loss')
ax1.plot(logs['epoch']+1, logs['out_class_accuracy'], '--', c='darkblue', label='Training accuracy')
ax1.plot(logs['epoch']+1, logs['val_out_class_accuracy'], '--', c='orange', label='Validation accuracy')
ax1.legend()

ax2.set_title('Regression')
ax2.plot(logs['epoch']+1, logs['out_reg_loss'], c='darkblue', label='Training loss')
ax2.plot(logs['epoch']+1, logs['val_out_reg_loss'], c='orange', label='Validation loss')
ax2.plot(logs['epoch']+1, logs['out_reg_MAE'], '--', c='darkblue', label='Training MAE')
ax2.plot(logs['epoch']+1, logs['val_out_reg_MAE'], '--', c='orange', label='Validation MAE')
ax2.legend()

plt.savefig(plot_path + 'training_curves.png')
