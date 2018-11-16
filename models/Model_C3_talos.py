import numpy as np
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Input, Dense, Flatten

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

    Y_train = Y_train.astype(np.float32).reshape((-1,1))
    Y_test = Y_test.astype(np.float32).reshape((-1,1))

    return X_train[0], X_test[0], Y_train, Y_test


def Params():
    params = """
kernel_size1 categorical {10,20,30,40} [10]
filters1 categorical {16,32,64,128} [32]
pool_size1 categorical {1,2,4,8} [1]
stride1 categorical {1,2} [1]

kernel_size2 categorical {10,20,30,40} [10]
filters2 categorical {16,32,64,128} [32]
pool_size2 categorical {1,2,4,8} [1]
stride2 categorical {1,2} [1]

kernel_size3 categorical {10,20,30,40} [10]
filters3 categorical {16,32,64,128} [32]
pool_size3 categorical {1,2,4,8} [1]
stride3 categorical {1,2} [1]

last_dense categorical {32,64,128,256} [64]"""

    params = {
        'kernel_size1': [10, 20, 30, 40],
        'filters1': [16, 32, 64, 128],
        'pool_size1': [1, 2, 4],
        'stride1': [1, 2],

        'kernel_size2': [10, 20, 30, 40],
        'filters2': [16, 32, 64, 128],
        'pool_size2': [1, 2, 4, 8],
        'stride2': [1, 2],

        'kernel_size3': [10, 20, 30, 40],
        'filters3': [16, 32, 64, 128],
        'pool_size3': [1, 2, 4, 8],
        'stride3': [1, 2],

        'last_dense': [32, 64, 128, 256]
    }
    return params


def POC_model(input_shape_hot, p):

    X_input1 = Input(shape=input_shape_hot)

    X = Conv1D(filters=int(p['filters1']), kernel_size=int(p['kernel_size1']), strides=1, activation='relu', kernel_initializer='he_uniform')(X_input1)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout']))(X)
    X = MaxPooling1D(pool_size=int(p['pool_size1']), strides=int(p['stride1']), padding='same')(X)

    X = Conv1D(filters=int(p['filters2']), kernel_size=int(p['kernel_size2']), strides=1, activation='relu', kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout']))(X)
    X = MaxPooling1D(pool_size=int(p['pool_size2']), strides=int(p['stride2']), padding='same')(X)

    X = Conv1D(filters=int(p['filters3']), kernel_size=int(p['kernel_size3']), strides=1, activation='relu', kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout']))(X)
    X = MaxPooling1D(pool_size=int(p['pool_size3']), strides=int(p['stride3']), padding='same')(X)

    X = Flatten()(X)
    X = Dense(int(p['last_dense']), activation='relu', kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout']))(X)

    X = Dense(1)(X)

    model = Model(inputs=[X_input1], outputs=X)

    return model
