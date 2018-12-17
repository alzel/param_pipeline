import numpy as np
import pandas as pd
import importlib.util
from keras.models import Model
from keras.layers import Conv1D,MaxPooling1D,GRU,BatchNormalization,Dropout,Input,Dense,Bidirectional,Flatten,Concatenate,Reshape
from hyperopt import hp

# Loading data and params
def return_paths():
    data_file = ['data/dataset_good_rsd2_merged.npz']
    
    models_l1_path = ['models/Model_C3_param.py',
                      'models/Model_C2_param.py',
                      'models/Model_C3_param.py',
                      'models/Model_C3_param.py']
    
    w_files = ['models/dataset_good_rsd2_prom_Model_C3_1234_4a7d6c72df3e960610ef7e9c61379575_best',
               'models/dataset_good_rsd2_5utr_Model_C2_1234_6854772ee390dc3a276480d3a6a45eb8_best',
               'models/dataset_good_rsd2_3utr_Model_C3_1234_64754c89862ffc02999d5427196326b4_best',
               'models/dataset_good_rsd2_term_Model_C3_1234_d61c8846e801f05d20916d6902cb87a7_best']
    
    return data_file, models_l1_path, w_files

def load_p(w_files):
    return [pd.read_csv(file+'.p',header=None,index_col=0)[1] for file in w_files]

def load_data_l1(fname):
    '''loads Xhot/Y input data for level1'''
    npzfile = np.load(fname)
    Xh_train = npzfile['arr_0']
    Xh_test = npzfile['arr_1']
    #Xv_train = npzfile['arr_2']
    #Xv_test = npzfile['arr_3']
    Y_train = npzfile['arr_4']
    Y_test = npzfile['arr_5']
    #X_train = list()
    #X_train.append(Xh_train)
    #X_train.append(Xv_train)
    #X_test = list()
    #X_test.append(Xh_test)
    #X_test.append(Xv_test)
    Y_train = Y_train.astype(np.float32).reshape((-1,1))
    Y_test = Y_test.astype(np.float32).reshape((-1,1))
    return Xh_train, Xh_test, Y_train, Y_test

def load_data(data_file):
    '''returns lists of data'''
    X_train, X_test, Y_train, Y_test = load_data_l1(data_file)
    limits = [[0,1000],[1000,1300],[1300,1650],[1650,2150]]
    X_train_list = []
    X_test_list = []
    for limit in limits:
        X_train_list.append(X_train[:,limit[0]:limit[1]])
        X_test_list.append(X_test[:,limit[0]:limit[1]])
    return X_train_list, X_test_list, Y_train, Y_test

def Params():
    '''sets hyperparam values or limits to consider'''
    params = {
        'gru_size1': [16, 32, 64, 128],
        'gru_size2': [16, 32, 64, 128],
        'gru_size3': [16, 32, 64, 128],
        'gru_size4': [16, 32, 64, 128],
        'gru_size5': [32, 64, 128, 256],
        'dropout1': (0, 1),
        'dropout2': (0, 1),
        'dropout3': (0, 1),
        'dropout4': (0, 1),
        'dropout5': (0, 1),
        'dropoutL': (0, 1),
        'denseL': [32, 64, 128, 256]
    }
    return {k: hp.choice(k, v) if type(v) == list else hp.uniform(k, v[0], v[1]) for k, v in params.items()}

# Loading and processing models
def load_module(model_path):
    '''loads module containing models given path'''
    spec = importlib.util.spec_from_file_location('module',model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_modules(model_paths):
    '''returns multiple modules'''
    modules = []
    for model_path in model_paths:
        modules.append(load_module(model_path))
    return modules

def set_trainable(models, true_false):
    '''sets trainable on all model layers true or false'''
    for model in models:
        for layer in model.layers:
            layer.trainable=true_false
    return models
    
def load_model_level1(module, w_file, Xshape, p):
    '''loads model from module with weights'''
    #X_train, X_test, Y_train, Y_test = module.load_data(data_file)
    model = module.POC_model(Xshape,p)
    model.load_weights(w_file)
    # pop last dense and flatten layers = last 5 layers in all models
    for i in range(5):
        model.layers.pop()
    #https://github.com/keras-team/keras/issues/2371
    #https://github.com/keras-team/keras/issues/3465
    # reconnect last left layer to output
    model.layers[-1].outbound_nodes = []
    model.outputs = [model.layers[-1].output]
    return model

def load_all_models_level1(modules, w_files, data_file, p):
    '''loads and returns list of models'''
    models = [] 
    Xshape = return_data_shapes(data_file[0])
    for i in range(len(w_files)):
        models.append(load_model_level1(modules[i], w_files[i], Xshape[i], p[i]))
    return models

def return_data_shapes(data_file):
    '''returns data shapes for model loading'''
    X_shapes = []
    X_train, X_test, Y_train, Y_test = load_data(data_file)
    for part in X_train:
        X_shapes.append(part.shape[1:3])
    return X_shapes

# define the ensemble l2
def POC_model(shapes, p):
    '''loads everything, also shapes so input doesnt matter'''
    # get stuff
    data_files, models_l1_path, w_files = return_paths()
    p_l1 = load_p(w_files)
    
    # load models
    modules = load_modules(models_l1_path)
    print(modules)
    models_l1 = load_all_models_level1(modules, w_files, data_files, p_l1)
    models_l1 = set_trainable(models_l1, False)

#     # define model input
    shapes = return_data_shapes(data_files[0])
    model_input = [Input(shape=shape) for shape in shapes]
    X = [models_l1[i](model_input[i]) for i in range(len(models_l1))]
#    last_layers = X
    
#     # will need to reshape proper layer outputs
#     last_shapes = [model.layers[-1].output_shape for model in models_l1]
#     dim3 = int(min(np.array(last_shapes)[:,2]))
#     last_layers = [Reshape([last_shapes[i][1]*int(last_shapes[i][2]/dim3),dim3])(last_layers[i]) for i in range(len(last_layers))]
#     X = Concatenate(axis=1)(last_layers)
    
    '''the model we add on top of preloaded level 1 models'''
    X1 = Bidirectional(GRU(p['gru_size1'], return_sequences=True, activation='relu', kernel_initializer='he_uniform'))(X[0])
    X1 = BatchNormalization()(X1)
    X1 = Dropout(float(p['dropout1']))(X1)
    
    X2 = Bidirectional(GRU(p['gru_size2'], return_sequences=True, activation='relu', kernel_initializer='he_uniform'))(X[1])
    X2 = BatchNormalization()(X1)
    X2 = Dropout(float(p['dropout2']))(X1)
    
    X3 = Bidirectional(GRU(p['gru_size3'], return_sequences=True, activation='relu', kernel_initializer='he_uniform'))(X[2])
    X3 = BatchNormalization()(X3)
    X3 = Dropout(float(p['dropout3']))(X3)
    
    X4 = Bidirectional(GRU(p['gru_size4'], return_sequences=True, activation='relu', kernel_initializer='he_uniform'))(X[3])
    X4 = BatchNormalization()(X4)
    X4 = Dropout(float(p['dropout4']))(X4)
    
    X = [Reshape([-1,16])(layer) for layer in [X1,X2,X3,X4]]
    X = Concatenate(axis=1)(X)
    
    X = Bidirectional(GRU(p['gru_size5'], return_sequences=False, activation='relu', kernel_initializer='he_uniform'))(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout5']))(X)   
    
    #X = Flatten()(X)    
    X = Dense(int(p['denseL']), activation='relu', kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropoutL']))(X)
    
    y = Dense(1)(X)
    
    model = Model(inputs = model_input, outputs = y, name='ensemble')
    
    return model
