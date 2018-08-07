
# coding: utf-8

# In[ ]:


# should go to model definition
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Conv1D,MaxPooling1D,LSTM,BatchNormalization,Dropout,Input,Dense,Bidirectional,Flatten,Concatenate
np.random.seed(1337)  # for reproducibility

def load_data(fname):
    # create some data
    X = np.linspace(-1, 1, 200)
    np.random.shuffle(X)    # randomize the data
    Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
    
    # plot data
    # plt.scatter(X, Y)
    # plt.show()

    X_train = list()
    X_test = list()
    
    # train test split
    X_train.append(X[:160])
    Y_train = Y[:160]     
    X_test.append(X[160:]) 
    Y_test = Y[160:].reshape((-1,1))    
    
    return X_train, X_test, Y_train, Y_test

def POC_model(input_shape_hot,DR):

    model = Sequential()
    model.add(Dense(output_dim=1, input_dim=1, use_bias=True))
    
    return model

