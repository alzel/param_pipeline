
# coding: utf-8

# In[1]:


# V8


# In[2]:


# IMPORT INTIAL LIBRARIES
#https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
#before importing any Keras modules !!

import numpy as np
import tensorflow as tf
import random

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

import os
#setting default path if interactive mode (run this cell only ONCE if in interactive mode)
if is_interactive():
    os.chdir("../")
    
import configparser
import sys
from os.path import basename


# In[3]:


# IMPORT VARIABLES

#change only THESE
model_path = "./models/THIS_model_input.py"
data_path = "./data/THIS_data.npz"


weights_dir = "./weights"
results_dir = "./results"
model_name = os.path.splitext(basename(model_path))[0]
weight_path = os.path.join(weights_dir, model_name)
csv_logger_path = os.path.join(results_dir, model_name + "_val_results.csv")
test_results_path = os.path.join(results_dir, model_name + "_test_results.csv")

for filename in [weight_path, csv_logger_path, test_results_path]:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
args = (sys.argv)
config_path = ""
if not is_interactive():
    model_path = args[1]
    config_path = args[2]
    weight_path = args[3]
    csv_logger_path = args[4]
    test_results_path = args[4] + "_testset"
    data_path = args[5]
    
#suffix = "{epoch:03d}-{val_loss:.3f}.hdf5"
suffix = "best.hdf5"
weight_model_path = "{}.{}".format(weight_path, suffix)
suffix2 = "last.hdf5"
weight_model_path_last = "{}.{}".format(weight_path, suffix2)


#hyperparameters   
ALPHA = 0.1 # learnrate
BETA = 0.999 # Adam
BETA2 = 0.99
EPSILON = 1e-6
MBATCH = 100 # batch size
DROPOUT = 0.1
EPOCHS = 3 # epochs
MIN_DELTA = 0.05
PATIENCE = 30
LRS_DROP = 0.5
LRS_EPOCH_DROP = 10
LRS_TRESHOLD = 20
LRS = False # learning rate scheduler activation
SHUFFLE = True
MODEL_LOAD = True
REPLICATE_SEED = 100

#loading from config 
if not is_interactive():
    config_file = args[2]
    config = configparser.ConfigParser()
    config.read(config_file)
    ALPHA = config.getfloat('main', 'alpha')
    BETA = config.getfloat('main', 'beta')
    BETA2 = config.getfloat('main', 'beta2')
    EPSILON = config.getfloat('main', 'epsilon')
    MBATCH = config.getint('main', 'mbatch')
    DROPOUT = config.getfloat('main', 'dropout')
    EPOCHS = config.getint('main', 'epochs')
    MIN_DELTA = config.getfloat('main', 'min_delta')
    PATIENCE = config.getint('main', 'patience')
    LRS_DROP = config.getfloat('main', 'lrs_drop')
    LRS_EPOCH_DROP = config.getint('main', 'lrs_epoch_drop')
    LRS_TRESHOLD = config.getint('main', 'lrs_treshold')    
    LRS = config.getboolean('main', 'LRS')
    SHUFFLE = config.getboolean('main', 'SHUFFLE')
    MODEL_LOAD = config.getboolean('main', 'MODEL_LOAD')
    REPLICATE_SEED = config.getint('main', 'REPLICATE_SEED')

#loading model difinitions    
model_file = open(model_path, 'r').read()
exec(model_file)


# In[4]:


# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

#import os
os.environ['PYTHONHASHSEED'] = str(REPLICATE_SEED+1)

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(REPLICATE_SEED+2)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

random.seed(REPLICATE_SEED+3)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

# potentially set this back to [2, 2]
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

#Previously set as:
# import keras.backend as K
# K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=2, 
#                                                    inter_op_parallelism_threads=2)))

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(REPLICATE_SEED+4)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Rest of code follows ...


# In[5]:


## ADDITIONAL LIBRARIES
import pandas as pd
from scipy import stats

if is_interactive():
    import matplotlib
    import pylab as plt
else:
    import matplotlib
    matplotlib.use('agg')
    import pylab as plt
    
from keras.models import Model, load_model
from keras.layers import Conv1D,MaxPooling1D,LSTM,BatchNormalization,Dropout,Input,Dense,Bidirectional,Activation,Flatten
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.backend import squeeze
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping, LearningRateScheduler
from keras.metrics import mean_squared_error


# In[6]:


#to display train and validation metrics on same tensorboard plot
class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./tensorboard_logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)
        
        # print learning rate for testing purposes
        # learning rate decay in optimizers changes internally and is not shown
        # https://stackoverflow.com/questions/37091751/keras-learning-rate-not-changing-despite-decay-in-sgd
        lr = float(K.get_value(self.model.optimizer.lr))
        print("Learning rate:", lr)
        
    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


# In[7]:


# define metrics and tests

# Keras backend implementations
def coef_det_k(y_true, y_pred): # order of variables defined in https://keras.io/backend/
    SS_res =  K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true-K.mean(y_true)))
    return 1-SS_res/(SS_tot+K.epsilon())

def corr_coef_k(y_true, y_pred):
    xm, ym = y_true-K.mean(y_true), y_pred-K.mean(y_pred)
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    return K.maximum(K.minimum(r_num/(r_den+K.epsilon()), 1.0), -1.0)

# numpy implementations
def mse_np(y_true, y_pred):
    return np.mean(np.square(y_true-y_pred))

def coef_det_np(y_true, y_pred):
    SS_res =  np.sum(np.square(y_true-y_pred))
    SS_tot = np.sum(np.square(y_true-np.mean(y_true)))
    return 1-SS_res/(SS_tot+1e-7)

def corr_coef_np(y_true, y_pred):
    return np.corrcoef(y_pred[:,0],y_true[:,0])[0,1]

# evaluations on test data 
def eval_on_test(X_test, Y_test, model, fname='', return_np=False):
    loss = model.evaluate(X_test, Y_test, X_test[0].shape[0])
    Y_pred = model.predict(X_test)
    
    x = Y_pred[:,0]
    y = Y_test[:,0]
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(x,y)
    plt.plot(x, y, 'o', label='original data')
    plt.plot(x, intercept + slope*x, 'r', label='fitted line')
    plt.legend()
    plt.title('R2 = {}'.format(loss[1]))
    plt.xlabel('Y_pred')
    plt.ylabel('Y_true')
    if len(fname)>0:
        plt.savefig(fname+'.pdf', bbox_inches='tight')
    
    loss.append(rvalue)
    
    if return_np:
        loss.append(mse_np(Y_test, Y_pred))
        loss.append(coef_det_np(Y_test, Y_pred))
        loss.append(corr_coef_np(Y_test, Y_pred))
    
    return loss


# In[8]:


#loading data
X_train, X_test, Y_train, Y_test = load_data(data_path)


# In[9]:


input_shape = X_train[0].shape[1:3]
model = POC_model(input_shape, DROPOUT)
model.summary()


# In[10]:


opt = Adam(lr=ALPHA, beta_1=BETA, beta_2=BETA2, epsilon=EPSILON, decay=0)
model.compile(loss='mse', optimizer=opt, metrics=[coef_det_k, corr_coef_k])
# lr_tmp = float(K.get_value(model.optimizer.lr))
# print(lr_tmp)


# In[11]:


# Set callbacks

# checkpoint
# https://machinelearningmastery.com/check-point-deep-learning-models-keras/
# https://keras.io/callbacks/ - for now save every epoch

check_best = ModelCheckpoint(weight_model_path, monitor='val_loss', verbose=0, 
                             save_best_only=True, save_weights_only=True, mode='auto')

check_last = ModelCheckpoint(weight_model_path_last, monitor='val_loss', verbose=0, 
                             save_best_only=False, save_weights_only=True, mode='auto')

# tensorboard
# http://fizzylogic.nl/2017/05/08/monitor-progress-of-your-keras-based-neural-network-using-tensorboard/

tensorboard = TrainValTensorBoard(write_graph=False, log_dir='./tensorboard_logs/' +                                   basename(data_path) + '_' + basename(csv_logger_path))

csv = CSVLogger(csv_logger_path, separator = ",", append = True)

# early stopping
earlystop = EarlyStopping(monitor='val_loss', min_delta=MIN_DELTA, patience=PATIENCE)

# learning rate scheduler
# add after certain amount of full training => treshold
# A typical way is to to drop the learning rate by half every 10 epochs.
# implements gradual decrease to drop by LRS_DROP every LRS_EPOCH_DROP epochs
def schedule(epoch, lr):
    treshold = LRS_TRESHOLD
    drop = LRS_DROP
    epoch_drop = LRS_EPOCH_DROP
    if epoch > treshold:
        lr *= pow(1-drop,1/epoch_drop)
    return float(lr)

if LRS:
    ALPHA = LearningRateScheduler(schedule)
    callbacks_list = [check_best, check_last, tensorboard, csv, earlystop, ALPHA]
else:
    callbacks_list = [check_best, check_last, tensorboard, csv, earlystop]

# in terminal run: tensorboard --logdir=logs/
# val_loss error is in callbacks, probably modelcheckpoint


# In[12]:


#checking if model exist then load best

if MODEL_LOAD:
    # import glob
    import re

    # def find_best_model(all_models):
    #         epochs = []
    #         losses = []
    #         for i, file in enumerate(all_models):
    #             groups = re.findall(weight_path + '.(.*)-(.*).hdf5', file)
    #             if groups:
    #                 epochs.append(int(groups[0][0]))
    #                 losses.append(float(groups[0][1]))
    #         return (all_models[np.argmin(losses)] )

    all_models = [os.path.join(os.path.dirname(weight_path), f)                   for f in os.listdir(os.path.dirname(weight_path))                   if re.match(os.path.basename(weight_path) + '\.'+suffix2, f)]

    if all_models:
        #best_model = find_best_model(all_models)
        best_model = all_models[0]
        print("Loading weights from {}".format(best_model))
        model.load_weights(best_model)


# In[13]:


# keras model checkpoint KeyError: 'val_loss'
# fix: https://github.com/keras-team/keras/issues/6104
# must add validation_split=xx

import time

start = time.time()
model.fit(X_train, Y_train, 
          batch_size=MBATCH, 
          epochs=EPOCHS, 
          validation_split=0.1, 
          shuffle=SHUFFLE, 
          callbacks=callbacks_list)

end = time.time()
print(end - start)


# In[14]:


# tests data on test set 
test_loss = eval_on_test(X_test, Y_test, model, fname=test_results_path)


# In[15]:


# Save test results
d = {'test_loss' : [test_loss[0]],
     'test_coef_det_k' : [test_loss[1]], 
     'test_corr_coef_k' : [test_loss[2]],
     'test_corr_coef_plot' : [test_loss[3]]}

test_df = pd.DataFrame(data=d)
test_df.to_csv(test_results_path, index=False)

