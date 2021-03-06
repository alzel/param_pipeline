import numpy as np
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Input, Dense, Flatten
from hyperopt import hp

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
        'filters1': [16],
        'pool_size1': [1],
#        'dropout1' : (0,1),
        'stride1': [1],
        'last_dense': [32]
#        'dropout2': (0, 1)
    }
    return {k: hp.choice(k, v) if type(v) == list else hp.uniform(k, v[0], v[1]) for k, v in params.items()}


def POC_model(input_shape, p):

    X_input1 = Input(shape=input_shape[0])

    X = Conv1D(filters=int(p['filters1']), kernel_size=int(p['kernel_size1']), strides=1, activation='relu', kernel_initializer='he_uniform')(X_input1)
    X = BatchNormalization()(X)
    X = Dropout(0.1)(X)
    X = MaxPooling1D(pool_size=int(p['pool_size1']), strides=int(p['stride1']), padding='same')(X)

    X = Flatten()(X)
    X = Dense(int(p['last_dense']),activation='relu',kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(0.1)(X)

    X = Dense(1)(X)
    model = Model(inputs=X_input1, outputs=X)

    return model
