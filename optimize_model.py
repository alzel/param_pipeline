from comet_ml import Optimizer, Experiment
import hashlib
import os
import random
import re
import inspect
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from my_utils import coef_det_k, best_check, last_check, TrainValTensorBoard, MyCSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping
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
                    required=True,
                    help="Directory to store model checkpoints")

parser.add_argument('--tensorboard_dir',
                    type=str,
                    default=None,
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
                    help="config file for parameter bounds in PCS format")

parser.add_argument('--validation_split', default=0.1, type=float,
                    help="Validation split")

parser.add_argument('--verbose', default=0, type=int, choices=[0, 1, 2],
                    help="Verbosity level of training")

parser.add_argument('--optimizer_iterations', default=10000, type=int,
                    help="Number of optimizer iterations")

parser.add_argument('--project_name', default="default", type=str,
                    help="Comet ML projectname")

parser.add_argument('--api_key',
                    required=True,
                    type=str,
                    help="Comet ML API KEY")

parser.add_argument('--workspace',
                    default='alzel',
                    type=str,
                    help="Comet ML workspace")

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

API_KEY = args.api_key
PROJECT_NAME = args.project_name
WORKSPACE = args.workspace


def split_data(x, y, validation_split=0.1):
    if validation_split and 0. < validation_split < 1.:
        split_at = int(len(x[:]) * (1. - validation_split))
        x, x_val = (x[0:split_at, :, :], x[split_at:, :, :])
        y, y_val = (y[0:split_at], y[split_at:])
    else:
        raise ValueError("validation_split must be [0,1)")
    return x, x_val, y, y_val


def wrapped_model(x=None, y=None, x_val=None, y_val=None, p=None):
    model = POC_model(x.shape[1:3], p)

    if args.multi_gpu and args.multi_gpu >= 2:
        model = multi_gpu_model(model, gpus=args.multi_gpu)

    print(p.params)

    model.compile(optimizer=Adam(lr=float(p['lr']), beta_1=float(p['beta_1']), beta_2=float(p['beta_2']),
                                 epsilon=float(p['epsilon'])), loss='mse', metrics=[coef_det_k])

    my_keys = sorted(list(suggestion.params.keys()))
    param_string = ','.join(["{!s}={!r}".format(key, suggestion[key]) for key in my_keys])
    hash_object = hashlib.md5(param_string.encode())

    #file_names for callbacks
    file_name = os.path.splitext(os.path.basename(args.output_file))[0]
    best_model_ckpt_file = os.path.join(args.model_ckpt_dir, file_name + hash_object.hexdigest()) + "_best"
    last_model_ckpt_file = os.path.join(args.model_ckpt_dir, file_name + hash_object.hexdigest()) + "_last"

    hpars = {k: np.array(suggestion.params[k][0]) for k in suggestion.params.keys()}
    hpars['md5'] = np.array(hash_object.hexdigest())

    call_backs = [EarlyStopping(monitor='val_loss', min_delta=float(p['min_delta']), patience=int(p['patience'])),
                  ModelCheckpoint(filepath=best_model_ckpt_file, **best_check),
                  ModelCheckpoint(filepath=last_model_ckpt_file, **last_check),
                  MyCSVLogger(filename=args.output_file, hpars=hpars, separator=",", append=True)]

    if args.tensorboard_dir:
        call_backs.append(TrainValTensorBoard(log_dir=os.path.join(args.tensorboard_dir, file_name + hash_object.hexdigest()),
                                              histogram_freq=10,
                                              write_grads=True))
    model.fit(x, y,
              batch_size=int(p['mbatch']),
              epochs=int(p['epochs']),
              verbose=args.verbose,
              validation_data=[x_val, y_val],
              callbacks=call_backs)

    return model


def create_hparams(file_path):
    """Loads generic training hyperparameters."""
    with open(file_path, "r") as pcs_file:
        pcs_content = pcs_file.read()
    return pcs_content


# loading model it's params and data
data_path = args.data
model_name = os.path.splitext(os.path.basename(args.model))[0]
exec("from models." + model_name + " import POC_model, load_data, Params")
X_train, X_test, Y_train, Y_test = load_data(data_path)


# splitting validation data
x, x_val, y, y_val = split_data(x=X_train, y=Y_train, validation_split=args.validation_split)

p_default = create_hparams(args.param_config)
p_specific = Params()
params = p_default + p_specific

# filters only defined parameters
lines = "\n".join([inspect.getsource(i) for i in [POC_model, wrapped_model]])
relevant_params = set(re.findall("p\['(\w+)'\]", lines))
params = "\n".join([line for line in params.splitlines() for k in relevant_params if re.search(r'\b{0}\b'.format(k), line)])

if args.CHUNKS:  # adding chunks

    if args.reverse:
        data_split_lst = [str(i[-1]) for i in np.array_split(range(x.shape[1]), args.CHUNKS)]
    else:
        data_split_lst = [str(i[0]) for i in np.array_split(range(x.shape[1]), args.CHUNKS)]

    data_split_param = "data_split categorical {" + ",".join(data_split_lst) + "}" + " [{}]\n".format(data_split_lst[0])
    params = params + '\n' + data_split_param

comet_optimizer = Optimizer(API_KEY)
comet_optimizer.set_params(params)

n = 0
while n < args.optimizer_iterations:
    print(f'Starting Iteration {n}')

    experiment = Experiment(
        api_key=API_KEY,
        workspace=WORKSPACE,
        project_name=PROJECT_NAME)

    suggestion = comet_optimizer.get_suggestion()

    if args.reverse:
        x_chunk = x[:, :int(suggestion['data_split']) + 1, :]
        x_val_chunk = x_val[:, :int(suggestion['data_split']) + 1, :]
        x_train_chunk = X_test[:, :int(suggestion['data_split']) + 1, :]
    else:
         x_chunk = x[:, int(suggestion['data_split']):, :]
         x_val_chunk = x_val[:, int(suggestion['data_split']):, :]
         x_train_chunk = X_test[:, int(suggestion['data_split']):, :]

    model = wrapped_model(x=x_chunk, x_val=x_val_chunk, y=y, y_val=y_val, p=suggestion)
    val_mse, val_coef_det = model.evaluate(x_val_chunk, y_val, batch_size=suggestion['mbatch'])


    # testing
    test_mse, test_coef_det = model.evaluate(x_train_chunk, Y_test, batch_size=suggestion['mbatch'])
    K.clear_session()  # because of tensorboard callbacks it dies otherwise

    metrics = {
        'val_mse': val_mse,
        'val_coef_det': val_coef_det,
        'test_mse': test_mse,
        'test_coef_det': test_coef_det
    }

    experiment.log_multiple_metrics(metrics)
    suggestion.report_score('val_mse', val_mse)

    print('Val mse:', val_mse)
    print('Val coef_det:', val_coef_det)
    print('Test mse:', test_mse)
    print('Test coef_det:', test_coef_det)

    n += 1
