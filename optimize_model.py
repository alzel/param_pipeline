import hashlib
import csv
import pickle
import os
import random
import re
import traceback
import inspect
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from my_utils import coef_det_k, best_check,last_check, TrainValTensorBoard, MyCSVLogger, create_hparams, split_data
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
from keras.utils import multi_gpu_model
from tqdm import tqdm
import argparse
import logging
from hyperopt import hp, tpe, fmin, Trials
from hyperopt import STATUS_OK, STATUS_FAIL

logging.basicConfig(format='%(asctime)s-%(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

#tensorflow configuration
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
config.log_device_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.45
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(description='Train my model.')
parser.add_argument('--model',
                    required=True,
                    type=str,
                    help='Model definition with hyperparameters dictionary')

parser.add_argument('--data',
                    required=True,
                    type=str,
                    help='Dataset')

parser.add_argument('--model_ckpt_dir',
                    type=str,
                    required=True,
                    default="model_weights",
                    help="Directory to store model checkpoints")

parser.add_argument('--tensorboard_dir',
                    type=str,
                    default=None,
                    help="Directory to store logs/tensorboard")

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


parser.add_argument('--validation_split', default=0.1, type=float,
                    help="Validation split")

parser.add_argument('--verbose', default=0, type=int, choices=[0, 1, 2],
                    help="Verbosity level of training")

parser.add_argument('--optimizer_iterations', default=1000, type=int,
                    help="Number of optimizer iterations")

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


def wrapped_model(p):

#     deletion lines 109-116
    x_chunk = x
    x_val_chunk = x_val

    model = POC_model(1, p) # should not matter as shapes loaded internally

    if args.multi_gpu and args.multi_gpu >= 2: # often crashes because of this
        model = multi_gpu_model(model, gpus=args.multi_gpu)

    sorted_param_keys = sorted(list(params.keys()))
    param_string = ','.join(["{!s}={!r}".format(key, p[key]) for key in sorted_param_keys])
    hash_string = hashlib.md5(param_string.encode()).hexdigest()

    #file_names for callbacks
    file_name = os.path.splitext(os.path.basename(args.output_file))[0]
    best_model_ckpt_file = os.path.join(args.model_ckpt_dir, file_name + "_" + hash_string) + "_best"
    last_model_ckpt_file = os.path.join(args.model_ckpt_dir, file_name + "_" + hash_string) + "_last"

    hpars = {k: np.array(p[k]) for k in p.keys()}
    hpars['id'] = np.array(hash_string)

    call_backs = [EarlyStopping(monitor='val_loss', min_delta=float(p['min_delta']), patience=int(p['patience'])),
                  ModelCheckpoint(filepath=best_model_ckpt_file, **best_check),
                  ModelCheckpoint(filepath=last_model_ckpt_file, **last_check),
                  MyCSVLogger(filename=args.output_file, hpars=hpars, separator=",", append=True)]

    if args.tensorboard_dir:
        call_backs.append(TrainValTensorBoard(log_dir=os.path.join(args.tensorboard_dir, file_name + hash_string),
                                              histogram_freq=10, write_grads=True))

    model.compile(optimizer=Adam(lr=p['lr'], beta_1=p['beta_1'], beta_2=p['beta_2'], epsilon=p['epsilon']),
                  loss='mse',
                  metrics=[coef_det_k])

    out = model.fit(x_chunk, y,
                    batch_size=int(p['mbatch']),
                    epochs=int(p['epochs']),
                    verbose=args.verbose,
                    validation_data=[x_val_chunk, y_val],
                    callbacks=call_backs)

    result = {
        'loss': min(out.history['val_loss']),
        'coef_det': max(out.history['val_coef_det_k']),
        'space': p,
        'history': out.history,
        'status': STATUS_OK
    }
    return result


def optimize_model(p):
    try:
        result = wrapped_model(p)
        K.clear_session()
        return result

    except Exception as err:
        try:
            K.clear_session()
        except:
            pass
        err_str = str(err)
        logging.error("Cannot optimize model", exc_info=True)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)

        return {
            'status': STATUS_FAIL,
            'err': err_str,
            'traceback': traceback_str
        }


def run_a_trial():
    """Run one TPE meta optimisation step and save its results."""
    max_evals = nb_evals = 1

    logging.info("Attempt to resume a past training if it exists:")
    file_name = os.path.join(os.path.dirname(args.output_file),
                             os.path.splitext(os.path.basename(args.output_file))[0] + ".pkl")
    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open(file_name, "rb"))
        logging.info("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        logging.info("Rerunning from {} trials to add another one.".format(len(trials.trials)))
    except:
        trials = Trials()
        logging.info("Starting from scratch: new trials.")

    best = fmin(
        optimize_model,
        params,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    try:
        pickle.dump(trials, open(file_name, "wb"))
        logging.info(f"Trial {len(trials.trials)} was saved")
    except Exception as err:
        logging.error("Cannot save trial", exc_info=True)

    return max_evals


if __name__ == "__main__":

    #loading parameters
    p_default = create_hparams(args.param_config)

    # loading model/data
    data_path = args.data
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    exec("from models." + model_name + " import POC_model, load_data, Params")
    try:
        X_train, X_test, Y_train, Y_test = load_data(data_path)
        logging.info("Successfully loaded data")
    except Exception as err:
        logging.error("Cannot load data", exc_info=True)

    # splitting validation data
    x, x_val, y, y_val = split_data(x=X_train, y=Y_train, validation_split=args.validation_split)

    p_specific = Params()
    params = {**p_default, **p_specific}

#     if args.CHUNKS:  # adding chunks
#         if args.reverse:
#             params['data_split'] = hp.choice('data_split', [i[-1] for i in np.array_split(range(x.shape[1]), args.CHUNKS)])
#         else:
#             params['data_split'] = hp.choice('data_split', [i[0] for i in np.array_split(range(x.shape[1]), args.CHUNKS)])

    # filters only defined parameters
    lines = "\n".join([inspect.getsource(i) for i in [POC_model, wrapped_model]])
    relevant_params = set(re.findall("p\['(\w+)'\]", lines))
    if 'id' in relevant_params:
        relevant_params.remove('id')
    params = {k: params[k] for k in relevant_params}


    pbar = tqdm(total=args.optimizer_iterations)
    n = 0
    #trying for the first time
    try:
        n = run_a_trial()
        pbar.update(n)
    except Exception as err:
        logging.error("Cannot run trial", exc_info=True)

    while n < args.optimizer_iterations:
        try:
            n = run_a_trial()
            pbar.update(1)
        except Exception as err:
            logging.error("Cannot run trial", exc_info=True)
            n += 1
            pbar.update(1)
