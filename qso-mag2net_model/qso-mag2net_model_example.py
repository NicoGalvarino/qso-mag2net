# Example of the model training for qso-mag2net 
# (sorry documentation currently very sparse and  code 
# not too pretty but should do for now)

# random seed
seed_value = 42

import os
#setting gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# setting random seeds
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)

# other imports
import h5py
import keras  # type: ignore
from keras.models import Sequential  # type: ignore
from keras.layers import Conv1D, MaxPooling1D, Dense, AveragePooling1D, Flatten  # type: ignore
from keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

# generator used for training
# import sys
# sys.path.append('../notebooks/')
from generators import generator_fiducial_model

# this part is for tf 1.x and deprecated
# translate for tf 2.x -> tf.config is used instead
# ------------------------------------------------------------------------------
gpu = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpu))

# from tensorflow.compat.v1 import ConfigProto  # config of the gpu session
# config = ConfigProto()
# config.gpu_options.allow_growth = True  # gpu memory used grows as needed
#                                         # instead of using the full gpu memory from the start
tf.config.experimental.set_memory_growth(gpu, True)

# from tensorflow.compat.v1 import InteractiveSession
# session = InteractiveSession(config=config)  # makes current session the default one
                                               # no need for passing a session obj to use tf operations
                                               # also runs tf operations immediately instead of default
                                               # by default added to a comp. graph and executed later
# eager execution and automatic session management by default in tf 2.x

# ------------------------------------------------------------------------------

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # ?

# decay of the learning rate
def step_decay(epoch):
    if epoch >=0 and epoch < 20:
        lrate = 0.001
    if epoch >= 20 and epoch < 80:
        lrate = 0.0001
    if epoch >= 80 and epoch < 120:
        lrate = 0.00001
    if epoch >= 120 and epoch <= 150:
        lrate = 0.000001
    return lrate

sample = h5py.File('./../../spectra_Roland/samples/training_samples/real_samples/WL_fixed/SNRfull_WLfull_range_fixed_ygap.hdf5', 'r')
# sample path -> training data only?

base_name = input('Name of the model ')  # save name of the model
model_path = './trained_model/'  # save path of the model
logger_path = './logfiles/'  # path to the log files to be created
checkpoint_path = './checkpoints/'  # path to the checkpoints to be created
dim_1 = 6316  # number of pixels in the spectra


# generator initial setup
dim = (dim_1, 1)
params_generator = {'dim': (dim_1,), 
                    'batch_size': 500,
                    'n_classes': 1,
                    'n_channels': 1,
                    'shuffle': True}

# model setup 
X_input = keras.Input(dim)
X = Conv1D(64, 5, activation='relu')(X_input)
X = AveragePooling1D(3)(X)
X = Conv1D(128, 10, activation='relu')(X)
X = AveragePooling1D(3)(X)
X = Conv1D(256, 10, activation='relu')(X)
X = Conv1D(256, 10, activation='relu')(X)
X = Conv1D(256, 10, activation='relu')(X)
X = AveragePooling1D(3)(X)
X = Flatten()(X)
X = Dense(512, activation='relu')(X)
X = Dense(512, activation='relu')(X)

# classification output
X_class_dense_out = Dense(1, activation='sigmoid', name='out_class')(X)

# regression output
X_reg_dense_out = Dense(1, activation='relu', name='out_reg')(X)

lr_scheduler = LearningRateScheduler(step_decay)
csv_logger = CSVLogger(logger_path + base_name + '_history.csv', append=True)
opt = Adam(0.001)

model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path+base_name, 
                                            monitor='val_loss',
                                            mode='min',
                                            save_weights_only=False,
                                            save_best_only=True)

model = keras.Model(inputs = X_input, outputs = [X_class_dense_out, X_reg_dense_out])
model.compile(optimizer=opt, 
              loss = {'out_reg':'MAE', 'out_class':'binary_crossentropy'}, 
              metrics = {'out_reg':'MAE', 'out_class':'accuracy'}, 
              loss_weights={'out_reg':1., 'out_class':300}  # why theses weights?
              )

print(model.summary())

sample_size = len(sample['flux'])

# training validation split and further generator setup
list_IDs_training = random.sample(range(sample_size), int(sample_size*0.8))  # 80% for training
list_IDs_validation = range(0, sample_size)

list_IDs_validation = np.setdiff1d(list_IDs_validation, list_IDs_training)

# training_generator = generator_WLfull.DataGenerator(list_IDs_training, sample, 'absorber_true', 
#                                                     'cent_WL_2796',  **params_generator)
# validation_generator = generator_WLfull.DataGenerator(list_IDs_validation, sample, 'absorber_true', 
#                                                       'cent_WL_2796', **params_generator)
training_generator = generator_fiducial_model.DataGenerator(list_IDs_training, sample, 'absorber_true', 
                                                            'cent_WL_2796',  **params_generator)
validation_generator = generator_fiducial_model.DataGenerator(list_IDs_validation, sample, 'absorber_true', 
                                                              'cent_WL_2796', **params_generator)

# model fitting
history = model.fit(training_generator, 
                    validation_data=validation_generator, 
                    epochs=150, 
                    verbose=1, 
                    shuffle=True, 
                    callbacks=[lr_scheduler, csv_logger, model_checkpoint_callback]
                    )

# saving model
model.save(model_path + base_name + '_model')
