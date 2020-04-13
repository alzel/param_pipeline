import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#for testing
from my_utils import coef_det_k, best_check,last_check, TrainValTensorBoard, MyCSVLogger, TestCallback, create_hparams, split_data
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.metrics import mse, binary_crossentropy, categorical_crossentropy
####

import numpy as np
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Input, Dense, Flatten
from hyperopt import hp
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)

def load_data(fname):
    # X is multi-variable array
    # Y contains single variable - fix shape for Keras

    npzfile = np.load(fname)
    Xh_train = npzfile['arr_0']
    Xh_test = npzfile['arr_1']
    Xv_train = npzfile['arr_2']
    Xv_test = npzfile['arr_3']
    Y_train = npzfile['arr_4']
    Y_test = npzfile['arr_5']

    X_train = list()
    X_train.append(Xh_train)
    X_train.append(Xv_train)
    X_test = list()
    X_test.append(Xh_test)
    X_test.append(Xv_test)

    Y_train = Y_train.astype(np.float32).reshape((-1, 1))
    Y_test = Y_test.astype(np.float32).reshape((-1, 1))

    return X_train[0], X_test[0], Y_train, Y_test


def Params():
    params = {
        'kernel_size1': [10],
        'filters1': [32],
        'pool_size1': [1],
#        'dropout1' : (0,1),
        'stride1': [1],
        'last_dense': [32]
#        'dropout2': (0, 1)
    }
    # return {k: hp.choice(k, v) if type(v) == list else hp.uniform(k, v[0], v[1]) for k, v in params.items()}
    return {k: v[0] if type(v) == list else v for k, v in params.items()}


def POC_model(input_shape, p):

    X_input1 = Input(shape=input_shape[0])

    X = Conv1D(filters=int(p['filters1']), kernel_size=int(p['kernel_size1']), strides=1, activation='relu', kernel_initializer='he_uniform')(X_input1)
    X = BatchNormalization()(X)
    #X = Dropout(0.1)(X)
    X = MaxPooling1D(pool_size=int(p['pool_size1']), strides=int(p['stride1']), padding='same')(X)

    X = Flatten()(X)
    X = Dense(int(p['last_dense']),activation='relu',kernel_initializer='he_uniform')(X)
    #X = BatchNormalization()(X)
    #X = Dropout(0.5)(X)

    X = Dense(1)(X)
    model = Model(inputs=X_input1, outputs=X)

    return model

# params/constants
LOSS = "mse"
METRICS = "coef_det_k"
DATA_FILE = "data/data_min-system-boxcox_corrected_rsd2.npz"

# defaults
p_default = {
    'epochs' : 10000,
    'min_delta' : 0.01,
    'patience' : 50,
    'lr' : 0.001,
    'beta_1' : 0.90,
    'beta_2' : 0.99,
    'epsilon' : 1e-10,
    'mbatch' : 64
}
p_specific = Params()
params = {**p_default, **p_specific}

p = params
X_train, X_test, Y_train, Y_test = load_data(DATA_FILE)

if not type(X_train) == list:
    X_train = [X_train]
    X_test = [X_test]

x, x_val, y, y_val = split_data(x=X_train, y=Y_train, validation_split=0.1)

x[0] = x[0][0:2,]
y = y[0:2,]

model = POC_model([sl.shape[1:] for sl in x], p)

model.compile(optimizer=Adam(lr=p['lr'], beta_1=p['beta_1'], beta_2=p['beta_2'], epsilon=p['epsilon']),

#model.compile(optimizer=RMSprop(lr=p['lr']),
              loss=eval(LOSS),
              metrics=[eval(METRICS)])

out = model.fit(x, y,
                batch_size=int(p['mbatch']),
                epochs=int(p['epochs']),
                verbose=2,
                validation_data=[x_val, y_val])
