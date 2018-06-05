# should go to model definition
import numpy as np
from keras.models import Model
from keras.layers import Conv1D,MaxPooling1D,LSTM,BatchNormalization,Dropout,Input,Dense,Bidirectional,Flatten

def load_data(fname):
    npzfile = np.load(fname)
    return npzfile['arr_0'],npzfile['arr_1'],npzfile['arr_2'],npzfile['arr_3'],npzfile['arr_4'],npzfile['arr_5']

def POC_model(input_shape_hot,DR):
    X_input1 = Input(shape = input_shape_hot)
    
    # L 1: CONV 
    X1 = Conv1D(filters=128, kernel_size=30, strides=1, activation='relu')(X_input1) # 620/1 + 1 = 621
    X1 = BatchNormalization()(X1)
    X1 = Dropout(DR)(X1)
    
    X1 = Flatten()(X1)
    
    X = Dense(64, activation='relu')(X1)
    X = BatchNormalization()(X)
    X = Dropout(DR)(X) 

    X = Dense(1)(X)

    model = Model(inputs = [X_input1], outputs = X)
    
    return model
