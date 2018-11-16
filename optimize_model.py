import talos as ta
import hashlib
import csv
import pickle
from collections.abc import Iterable
from ruamel.yaml import YAML
import os
import random
import re
import json
import inspect
import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams
from keras.optimizers import Adam
from my_utils import coef_det_k, best_check,last_check, TrainValTensorBoard, MyCSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.utils import multi_gpu_model

import argparse

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(description='Train my model.')
parser.add_argument('-m',
                    '--model',
                    required=True,
                    type=str,
                    help='Model definition with hyperparameters dictionary')
parser.add_argument('-d',
                    '--data',
                    required=True,
                    type=str,
                    help='Dataset')

parser.add_argument('--model_ckpt_dir',
                    type=str,
                    default="model_weights",
                    help="Directory to store model checkpoints")

parser.add_argument('--tensorboard_dir',
                    type=str,
                    default="tensorboard",
                    help="Directory to store logs/tensorboard")

parser.add_argument('--experiment_no',
                    type=str,
                    default="1",
                    help="name of experiment")

parser.add_argument('--CHUNKS',
                    type=int,
                    default=1,
                    help="Chunks")
parser.add_argument('--reverse',
                    type=bool,
                    default=False,
                    help="reverses X_train, X_test sequences")
parser.add_argument('--REPLICATE_SEED',
                    type=int,
                    default=123,
                    help="SEED number")
parser.add_argument('--multi_gpu',
                    type=int,
                    default=None,
                    choices=[1,2,3,4],
                    help="Specify number of GPU in multi_gpu_model")

parser.add_argument('--param_config',
                    type=str,
                    help="config file for parameter bounds")

parser.add_argument('--hparams', type=str,
                    help='Comma separated list of "name=value" pairs of model hyper parameters.')

parser.add_argument('--hashmap', type=str,
                    help='Save hashes')

parser.add_argument('--results', type=str,
                    required=True, help="Results filename of talos.Scan object")

parser.add_argument('--val_split', default=0.1, type=float,
                    help="Validation split")

parser.add_argument('--grid_downsample', default=0.001, type=float,
                    help="Grid downsample, percent of full grid to sample")

parser.add_argument('--verbose', default=0, type=int, choices=[0, 1, 2],
                    help="Verbosity level of training")

parser.add_argument('--output_file',
                    required=True,
                    help="Output filename")

args = parser.parse_args()


# callback directories
if args.model_ckpt_dir:
    os.makedirs(args.model_ckpt_dir, exist_ok=True)

if args.tensorboard_dir:
    os.makedirs(args.tensorboard_dir, exist_ok=True)

# setting seeds
REPLICATE_SEED = args.REPLICATE_SEED
os.environ['PYTHONHASHSEED'] = str(REPLICATE_SEED + 1)
np.random.seed(REPLICATE_SEED + 2)
random.seed(REPLICATE_SEED + 3)
tf.set_random_seed(REPLICATE_SEED + 4)

def split_data(x, y, validation_split=0.1):
    if validation_split and 0. < validation_split < 1.:
        split_at = int(len(x[:]) * (1. - validation_split))
        x, x_val = (x[0:split_at, :, :], x[split_at:, :, :])
        y, y_val = (y[0:split_at], y[split_at:])
    else:
        raise ValueError("validation_split must be [0,1)")
    return x, x_val, y, y_val


def get_section_hparams(ub, lb, n):
    # for alphas, epsilon
    if isinstance(n, int) and (isinstance(ub, float) or isinstance(lb, float)) and \
            ub - lb <= 0.1 and lb > 0:
        prams = 1 - 10 ** np.random.uniform(np.log10(1 - lb), np.log10(1 - ub), n)
    # for betas, betas2
    elif isinstance(n, int) and (isinstance(ub, float) or isinstance(lb, float)) and \
            ub - lb <= 1 and lb > 0:
        prams = 10 ** np.random.uniform(np.log10(lb), np.log10(ub), n)
    # for dropouts
    elif isinstance(n, float):
        prams = np.arange(lb, ub + 0.0001, n)
    else:
        # for etc
        precision = 0 if ub > 1 else 2
        prams = np.round(np.random.uniform(lb, ub, n), precision)
    return prams.tolist()


def wrapped_model(x_train, y_train, x_val, y_val, p):

    if args.reverse:
        x_train = x_train[:, :int(p['data_split']) + 1, :]
        x_val= x_val[:, :int(p['data_split']) + 1, :]
        #x_train_chunk = X_test[:, :int(suggestion['data_split']) + 1, :]
    else:
         x_train = x_train[:, int(p['data_split']):, :]
         x_val = x_val[:, int(p['data_split']):, :]
         #x_train_chunk = X_test[:, int(suggestion['data_split']):, :]

    model = POC_model(x_train.shape[1:3], p)

    if args.multi_gpu and args.multi_gpu >= 2:
        model = multi_gpu_model(model, gpus=args.multi_gpu)

    my_keys = sorted(list(p.keys()))
    param_string = ','.join(["{!s}={!r}".format(key, p[key]) for key in my_keys])
    hash_object = hashlib.md5(param_string.encode())

    #file_names for callbacks
    file_name = os.path.splitext(os.path.basename(args.output_file))[0]
    best_model_ckpt_file = os.path.join(args.model_ckpt_dir, file_name + hash_object.hexdigest()) + "_best"
    last_model_ckpt_file = os.path.join(args.model_ckpt_dir, file_name + hash_object.hexdigest()) + "_last"

    print(p)

    hpars = {k: np.array(p[k]) for k in p.keys()}
    print(hpars)

    hpars['md5'] = np.array(hash_object.hexdigest())

    call_backs = [EarlyStopping(monitor='val_loss', min_delta=float(p['min_delta']), patience=int(p['patience'])),
                  ModelCheckpoint(filepath=best_model_ckpt_file, **best_check),
                  ModelCheckpoint(filepath=last_model_ckpt_file, **last_check),
                  MyCSVLogger(filename=args.output_file, hpars=hpars, separator=",", append=True)]

    if args.tensorboard_dir:
        call_backs.append(TrainValTensorBoard(log_dir=os.path.join(args.tensorboard_dir, file_name + hash_object.hexdigest()),
                                              histogram_freq=10, write_grads=True))

    model.compile(optimizer=Adam(lr=p['lr'], beta_1=p['beta_1'], beta_2=p['beta_2'], epsilon=p['epsilon']),
                  loss='mse',
                  metrics=[coef_det_k])

    if p['lrs']:
        def schedule(epoch, lr):
            treshold = p['lrs_threshold']
            drop = p['lrs_drop']
            epoch_drop = p['lrs_epoch_drop']
            if epoch > treshold:
                lr *= pow(1 - drop, 1 / epoch_drop)
            return float(lr)

        call_backs.append(LearningRateScheduler(schedule))

    out = model.fit(x_train, y_train,
                    batch_size=int(p['mbatch']),
                    epochs=int(p['epochs']),
                    verbose=args.verbose,
                    validation_data=[x_val, y_val],
                    callbacks=call_backs)
    return out, model


def create_hparams():
    """Create the hparams object for generic training hyperparameters."""
    hparams = HParams(
        mbatch=[256],
        lr=[0.1],
        epochs=200,
        dropout=[0.40])

    # Config file overrides any of preceding hyperparameter values
    if args.param_config:
        with open(args.param_config) as fp:
            cfg = YAML().load(fp)

            section = 'default'
            if section in cfg:
                for k, v in cfg[section].items():
                    if hparams.__contains__(k):
                        hparams.set_hparam(k, v)
                    else:
                        hparams.add_hparam(k, v)
            section = 'sampling'
            schema = ['ub', 'lb', 'n']
            if section in cfg:
                for k in cfg[section].keys():
                    if isinstance(cfg[section][k], Iterable) and all(s in cfg[section][k] for s in schema):
                        v = get_section_hparams(ub=cfg[section][k]['ub'],
                                                lb=cfg[section][k]['lb'],
                                                n=cfg[section][k]['n'])
                        if hparams.__contains__(k):
                            hparams.set_hparam(k, v)
                        else:
                            hparams.add_hparam(k, v)
    # Command line flags override any of the preceding hyperparameter values.
    if args.hparams:
        hparams = hparams.parse(args.hparams)
    json_string = hparams.to_json()
    prms = json.loads(json_string)
    for k, v in prms.items():
        prms[k] = list(np.atleast_1d(v))

    return prms


p_default = create_hparams()
# loading model
data_path = args.data
model_name = os.path.splitext(os.path.basename(args.model))[0]
exec("from models." + model_name + " import POC_model, load_data, Params")
X_train, X_test, Y_train, Y_test = load_data(data_path)

p_specific = Params()
params = {**p_default, **p_specific}

if args.CHUNKS:  # adding chunks
    if args.reverse:
        params['data_split'] = [i[-1] for i in np.array_split(range(X_train.shape[1]), args.CHUNKS)]
    else:
        params['data_split'] = [i[0] for i in np.array_split(range(X_train.shape[1]), args.CHUNKS)]

# filters only defined parameters
lines = "\n".join([inspect.getsource(i) for i in [POC_model, wrapped_model]])
relevant_params = set(re.findall("p\['(\w+)'\]", lines))
params = {k: params[k] for k in relevant_params}


dataset_name = os.path.splitext(os.path.basename(args.data))[0] + "_" + \
               os.path.splitext(os.path.basename(args.model))[0]


h = ta.Scan(X_train, Y_train,
            params=params,
            model=wrapped_model,
            dataset_name=args.output_file,
            experiment_no=args.experiment_no,
            val_split=args.val_split,
            grid_downsample=args.grid_downsample
            )

# saving results
with open(args.results, "wb") as f:
    pickle.dump(h, f, pickle.HIGHEST_PROTOCOL)
