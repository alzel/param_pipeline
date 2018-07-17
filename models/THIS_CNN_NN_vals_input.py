# should go to model definition
import numpy as np
from keras.models import Model
from keras.layers import Conv1D,MaxPooling1D,LSTM,BatchNormalization,Dropout,Input,Dense,Bidirectional,Flatten,Concatenate

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

    return X_train, X_test, Y_train, Y_test

def POC_model1(input_shape_hot,DR):

    X_input1 = Input(shape = input_shape_hot)

    # L 1: CONV
    X1 = Conv1D(filters=32, kernel_size=10, strides=1, activation='relu')(X_input1) # 620/1 + 1 = 621
    X1 = BatchNormalization()(X1)
    X1 = Dropout(DR)(X1)

    X1 = MaxPooling1D(pool_size=4, strides=4)(X1)

    X1 = Conv1D(filters=64, kernel_size=10, strides=1, activation='relu')(X_input1) # 611/1 + 1 = 612
    X1 = BatchNormalization()(X1)
    X1 = Dropout(DR)(X1)

    X1 = MaxPooling1D(pool_size=4, strides=4)(X1)

    X1 = Conv1D(filters=128, kernel_size=10, strides=1, activation='relu')(X_input1) # 602/1 + 1 = 603
    X1 = BatchNormalization()(X1)
    X1 = Dropout(DR)(X1)

    X1 = MaxPooling1D(pool_size=4, strides=4)(X1)

    X1 = Flatten()(X1)

    X = Dense(64, activation='relu')(X1)
    X = BatchNormalization()(X)
    X = Dropout(DR)(X)

    X = Dense(1)(X)

    model = Model(inputs = [X_input1], outputs = X)

    return model

def POC_model2(input_shape_hot,input_shape_val,DR):

    X_input1 = Input(shape = input_shape_hot)
    X_input2 = Input(shape = input_shape_val)

    # L 1: CONV
    X1 = Conv1D(filters=32, kernel_size=10, strides=1, activation='relu')(X_input1) # 620/1 + 1 = 621
    X1 = BatchNormalization()(X1)
    X1 = Dropout(DR)(X1)

    X1 = MaxPooling1D(pool_size=4, strides=4)(X1)

    X1 = Conv1D(filters=64, kernel_size=10, strides=1, activation='relu')(X_input1) # 611/1 + 1 = 612
    X1 = BatchNormalization()(X1)
    X1 = Dropout(DR)(X1)

    X1 = MaxPooling1D(pool_size=4, strides=4)(X1)

    ### Due to mistake only takes from here

    X1 = Conv1D(filters=128, kernel_size=10, strides=1, activation='relu')(X_input1) # 602/1 + 1 = 603
    X1 = BatchNormalization()(X1)
    X1 = Dropout(DR)(X1)

    X1 = MaxPooling1D(pool_size=4, strides=4)(X1)

    X1 = Flatten()(X1)

    ### above transplant and freeze

    X1 = Concatenate(axis=1)([X1,X_input2])

    X = Dense(64, activation='relu')(X1)
    X = BatchNormalization()(X)
    X = Dropout(DR)(X)

    X = Dense(1)(X)

    model = Model(inputs = [X_input1,X_input2], outputs = X)

    return model


def POC_model(input_shape_hot, DR):
    model1 = POC_model1(input_shape_hot, DR)
    model2 = POC_model2(input_shape_hot, X_train[1].shape[1:3], DR)
    print("Loading model from disk..")
    model1.load_weights("./misc/POC_v1_bdgh_0.02704_0.98673_0.1_512.500-0.699.hdf5")

    for i in range(0,5):
        model2.layers[i].set_weights(model1.layers[i].get_weights())

    model2.layers[0].trainable = True
    model2.layers[1].trainable = True
    model2.layers[2].trainable = True
    model2.layers[3].trainable = True
    model2.layers[4].trainable = True
    model2.layers[5].trainable = True

    del model1
    return model2
