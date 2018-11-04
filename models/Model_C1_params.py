import numpy as np
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D,  BatchNormalization, Dropout, Input, Dense, Flatten

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
    p_specific = {
        'filters1': [16, 32, 64, 128],
        'kernel_size1': [10, 20, 30, 40],
        'pool_size1': [1, 16, 32, 64, 128],
        'last_dense': [32, 64, 128, 256],
    }
    return p_specific


def POC_model(x_train, y_train, x_val, y_val, p):

    X_input1 = Input(shape=x_train.shape[1:3])
    X = Conv1D(filters=int(p['filters1']),
               kernel_size=int(p['kernel_size1']),
               strides=1,
               activation='relu',
               kernel_initializer='he_uniform')(X_input1)
    X = BatchNormalization()(X)
    X = Dropout(p['dropout'])(X)
    X = MaxPooling1D(pool_size=int(p['pool_size1']),
                     strides=int(p['pool_size1'] / 2) if p['pool_size1'] != 1 else None, padding='same')(X)

    X = Flatten()(X)
    X = Dense(int(p['last_dense']),
              activation='relu',
              kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(p['dropout'])(X)

    X = Dense(1)(X)
    model = Model(inputs=[X_input1], outputs=X)

    return model
