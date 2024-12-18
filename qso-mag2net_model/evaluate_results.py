import tensorflow as tf
import numpy as np
import h5py
import pandas as pd

from sklearn.metrics import (classification_report, accuracy_score, precision_score, recall_score, f1_score, 
                            mean_absolute_error, confusion_matrix, roc_curve, auc)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
import mpl_scatter_density
import seaborn as sns

# ---------------------------------------------------------------------------------------------------------------------------------------
# read test data
test_set_ = h5py.File('./../../spectra_Roland/samples/test_samples/real_samples/SNRfull_WLfull_WL_fixed_range_fixed_test_ygap.hdf5', 'r')
# 'EW_r_2796', 'EW_r_2803', 'EW_r_total', 'SNR', 'absorber_true', 'cent_WL_2796', 'cent_pix_2796', 'flux', 'master_wave', 'redshift'

X_test = test_set_['flux'][:]  # spectra
y_test_class = test_set_['absorber_true'][:]  # classification labels
y_test_reg = test_set_['cent_WL_2796'][:]  # regression labels

# ---------------------------------------------------------------------------------------------------------------------------------------
# read predictions
model_path = './checkpoints/first_attempt_Rolspec_5_epochs_526_batchsize/'

y_pred_class_prob = np.load(model_path + 'y_pred_class_prob.npy')
y_pred_class = np.load(model_path + 'y_pred_class.npy')
y_pred_reg = np.load(model_path + 'y_pred_reg.npy')

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
# plot classification

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
fig_c.savefig(model_path + '/results_classification.png', dpi=400)

# plot regression ----------------------------------------------------------------
# lambda true vs pred

fig_r = plt.figure(figsize=(10, 5))
ax = fig_r.add_subplot(1, 1, 1, projection='scatter_density')

white_plasma = LinearSegmentedColormap.from_list('white_plasma', 
                                                 ['white'] + plt.cm.plasma(np.linspace(0, 1, 256)).tolist())

density_range = np.histogram2d(y_test_reg[~np.isnan(y_test_reg) & ~np.isnan(y_pred_reg)], 
                               y_pred_reg[~np.isnan(y_test_reg) & ~np.isnan(y_pred_reg)]
                               )[0]
vmin = max(1, density_range[density_range > 0].min())
vmax = density_range.max()
print(vmin, vmax)

density = ax.scatter_density(y_test_reg[~np.isnan(y_test_reg) & ~np.isnan(y_pred_reg)], 
                             y_pred_reg[~np.isnan(y_test_reg) & ~np.isnan(y_pred_reg)], 
                             cmap=white_plasma, # vmin=0, 
                             dpi=72, norm=LogNorm(vmin=vmin, vmax=vmax))

print('density ok')
fig_r.colorbar(density, label='Count (log scale)')
# ax.plot([y_test_reg[~np.isnan(y_test_reg) & ~np.isnan(y_pred_reg)].min(), 
#          y_test_reg[~np.isnan(y_test_reg) & ~np.isnan(y_pred_reg)].max()], 
#         [y_test_reg[~np.isnan(y_test_reg) & ~np.isnan(y_pred_reg)].min(), 
#          y_test_reg[~np.isnan(y_test_reg) & ~np.isnan(y_pred_reg)].max()], 
#         lw=1, c='k', alpha=0.4)

ax.set_xlabel(r'True $\lambda$')
ax.set_ylabel(r'Predicted $\lambda$')
ax.set_title(r'True vs. Predicted $\lambda$')

fig_r.tight_layout()
fig_r.savefig(model_path + '/results_regression.png', dpi=400)
