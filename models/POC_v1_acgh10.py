
# coding: utf-8

# In[ ]:


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
    #X_train.append(Xv_train)
    X_test = list()
    X_test.append(Xh_test)
    #X_test.append(Xv_test)
    
    Y_train = Y_train.astype(np.float32).reshape((-1,1))
    Y_test = Y_test.astype(np.float32).reshape((-1,1))

    return X_train, X_test, Y_train, Y_test

def POC_model(input_shape_hot,DR):

    X_input1 = Input(shape = input_shape_hot)
    
    # L 1: CONV 
    X1 = Conv1D(filters=128, kernel_size=10, strides=1, activation='relu')(X_input1) # 640/1 + 1 = 641
    X1 = BatchNormalization()(X1)
    X1 = Dropout(DR)(X1)
    
    X1 = Flatten()(X1)
    
    X = Dense(64, activation='relu')(X1)
    X = BatchNormalization()(X)
    X = Dropout(DR)(X) 

    X = Dense(1)(X)

    model = Model(inputs = [X_input1], outputs = X)
    
    return model

