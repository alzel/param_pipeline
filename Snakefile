# V8: pure randomness

import re
import os
import glob
import numpy as np
configfile: "config.yaml"

localrules: all, makeDefault, makeConfigs

# number of seeds defines number of replicates, seed given in filenames
N = config['params']['configs_n']
REPLICATE_SEED = config['params']['replicate_seed']

# parameter limits
# log
alphaH = config['params']['alphaH']
alphaL = config['params']['alphaL']
betaH = config['params']['betaH']
betaL = config['params']['betaL']
beta2H = config['params']['beta2H']
beta2L = config['params']['beta2L']
epsilonH = config['params']['epsilonH']
epsilonL = config['params']['epsilonL']
# lin
mbatchL = config['params']['mbatchH']
mbatchH = config['params']['mbatchL']
dropoutL = config['params']['dropoutH']
dropoutH = config['params']['dropoutL']
lrs_tresholdL = config['params']['lrs_tresholdH']
lrs_tresholdH = config['params']['lrs_tresholdL']
lrs_dropL = config['params']['lrs_dropH']
lrs_dropH = config['params']['lrs_dropL']

# generate hyperparameter combinations
ALPHA = np.zeros((N,1))
BETA = np.zeros((N,1))
BETA2 = np.zeros((N,1))
EPSILON = np.zeros((N,1))
MBATCH = np.zeros((N,1))
DROPOUT = np.zeros((N,1))
LRS_TRESHOLD = np.zeros((N,1))
LRS_DROP = np.zeros((N,1))
PARAMS = []
seps = ':'

for i in range(N):
    LRS_DROP[i] = np.round(np.random.uniform(lrs_dropL, lrs_dropH),2)
    LRS_TRESHOLD[i] = np.int(np.round(np.random.uniform(lrs_tresholdL, lrs_tresholdH),0))
    DROPOUT[i] = np.round(np.random.uniform(dropoutL, dropoutH),2)
    MBATCH[i] = np.int(2**np.round(np.random.uniform(np.log2(mbatchL), np.log2(mbatchH))))
    EPSILON[i] = 10**np.random.uniform(np.log10(np.float(epsilonL)), np.log10(np.float(epsilonH)))
    BETA2[i] = 1-10**np.random.uniform(np.log10(1-beta2L), np.log10(1-beta2H))
    BETA[i] = 1-10**np.random.uniform(np.log10(1-betaL), np.log10(1-betaH))
    ALPHA[i] = 10**np.random.uniform(np.log10(np.float(alphaL)), np.log10(np.float(alphaH)))

    # each combination is a specific configuration
    PARAMS.append((str(ALPHA[i])[1:-1]+seps+
                str(BETA[i])[1:-1]+seps+
                str(BETA2[i])[1:-1]+seps+
                str(EPSILON[i])[1:-1]+seps+
                str(MBATCH[i])[1:-2]+seps+
                str(DROPOUT[i])[1:-1]+seps+
                str(LRS_TRESHOLD[i])[1:-2]+seps+
                str(LRS_DROP[i])[1:-1]).replace(" ", ""))

MODELS =  expand("results/{dataset}/{model}_{params}_{replicate_seed}.csv",
                 dataset=config['input_files']['datasets'],
                 model=config['input_files']['models'],
                 params=PARAMS,
                 replicate_seed=REPLICATE_SEED)

# CONFIGS =  expand("configs/config_{params}_{replicate_seed}.cfg",
#                  #dataset=config['input_files']['datasets'],
#                  #model=config['input_files']['models'],
#                  params=PARAMS,
#                  replicate_seed=REPLICATE_SEED)

if not os.path.exists("logs"):
    os.makedirs("logs")

rule all:
    input:
        MODELS
        #CONFIGS

#saving defualt params
rule makeDefault:
    input:
        "config.yaml"
    output:
        "defaults/config_default.cfg"
    run:
        import configparser
        import yaml
        config = ""
        print(input)
        with open(input[0], 'r') as stream:
            try:
                config = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        print (config)
        if config:
            with open(output[0], 'w') as configfile:

                config_new = configparser.RawConfigParser()
                config_new.add_section('main')
                config_new.set('main', "SHUFFLE" , config['params']['shuffle'] )
                config_new.set('main', "MODEL_LOAD" , config['params']['model_load'] )
                config_new.set('main', "LRS" , config['params']['lrs'] )
                config_new.set('main', "LRS_EPOCH_DROP" , config['params']['lrs_epoch_drop'] )
                config_new.set('main', "EPOCHS" , config['params']['epochs'])
                config_new.set('main', "MIN_DELTA" , config['params']['min_delta'])
                config_new.set('main', "PATIENCE" , config['params']['patience'])
                config_new.write(configfile)

rule makeConfigs:
    input:
        "defaults/config_default.cfg"
    output:
        "configs/config_{params}_{replicate_seed}.cfg"
    run:
        import configparser
        config_default = configparser.ConfigParser()
        config_default.read(input)

        for file in output:
            with open(file, 'w') as configfile:
                #with open(file, 'w') as out:

                config = configparser.ConfigParser()

                for section in config_default.sections():
                    if not config.has_section(section):
                        config.add_section(section)
                    for key, value in config_default.items(section):
                        config.set(section, key, value)

                tmp = wildcards.params.split(seps)
                config.set('main', "ALPHA" , tmp[0])
                config.set('main', "BETA" , tmp[1])
                config.set('main', "BETA2" , tmp[2])
                config.set('main', "EPSILON" , tmp[3])
                config.set('main', "MBATCH" , tmp[4])
                config.set('main', "DROPOUT" , tmp[5])
                config.set('main', "LRS_TRESHOLD" , tmp[6])
                config.set('main', "LRS_DROP" , tmp[7])
                config.set('main', "REPLICATE_SEED" , wildcards.replicate_seed)
                config.write(configfile)

#making directoris for params, whose are not automatic
import os
for dataset in config['input_files']['datasets']:
    os.makedirs(os.path.join("weights", dataset), exist_ok=True)

rule run_model:
    input:
        template=config["input_files"]["template"],
        dataset=lambda wildcards: config["input_files"]["datasets"][wildcards.dataset],
        model=lambda wildcards: config["input_files"]["models"][wildcards.model],
        config="configs/config_{params}_{replicate_seed}.cfg"
    output:
        results="results/{dataset}/{model}_{params}_{replicate_seed}.csv"
    params:
        weights="weights/{dataset}/{model}_{params}_{replicate_seed}"
    log:
        log1="logs/{dataset}/{model}_{params}_{replicate_seed}.log"
    run:
        import socket
        import getpass
        import os
        prefix = ""
        #hacks needed to run different tensoflows depending if node has GPU
        if "hebbe24-7" in socket.gethostname() or "hebbe24-5" in socket.gethostname():
            prefix = ("module load Anaconda3; module load GCC/6.4.0-2.28 OpenMPI/2.1.2; module load TensorFlow/1.6.0-Python-3.6.4-CUDA-9.1.85; source activate python364;")
        elif "hebbe" in socket.gethostname():
            prefix = ("module load Anaconda3; module load GCC/6.4.0-2.28 OpenMPI/2.1.2; module load TensorFlow/1.6.0-Python-3.6.4; source activate python364;")

        #particular configuration for my MacBook pro
        if "liv003l" in socket.gethostname():
            prefix = ("source activate tensorflow36;")

        import re
        template = input.template
        os.system('jupyter nbconvert --to=python ' + template)
        template = re.sub(".ipynb", "", template, count=0, flags=0)
        command = "python {} {} {} {} {} {} 2>&1 | tee {} ".format(template + '.py', input.model, input.config, params.weights, output.results, input.dataset, log.log1)
        print (command)
        os.system(prefix + command)
        #python {input.template} {input.model} {input.config} {params.weights} {output.results} #sys.arg[1] - model name;  #sys.arg[2] - config;  sys.arg[3] - optimization results;
