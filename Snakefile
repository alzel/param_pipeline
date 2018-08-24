# V7: adds replicates at fixed param settings

import re
import os
import glob
import numpy as np
configfile: "config.yaml"

localrules: all, makeDefault, makeConfigs

np.random.seed(123)
###generating parameters
#alpha very important for sure 1
#a = config['params']['alpha']['a']
#n = config['params']['alpha']['n']
#r = np.around(a*np.random.rand(n), 3)
ALPHA = config['params']['alpha'] #ALPHA = np.unique(np.around(10 ** r, 5))

#beta 1
#beta = 1 - 10 ^ r
#r[-a, -b] e.g. r E [-3, -1]
#a = config['params']['beta']['a']
#b = config['params']['beta']['b']
#n = config['params']['beta']['n']
#r = (a+1)*np.random.rand(n)
BETA = config['params']['beta'] #BETA = np.unique(np.around(1 - 10 ** (r + b), 5))

#beta 2
#beta = 1 - 10 ^ r
#r[-a, -b] e.g. r E [-3, -1]
#a = config['params']['beta2']['a']
#b = config['params']['beta2']['b']
#n = config['params']['beta2']['n']
#r = (a+1)*np.random.rand(n)
BETA2 = config['params']['beta2'] #BETA2 = np.unique(np.around(1 - 10 ** (r + b), 5))
EPSILON = config['params']['epsilon']

# decay can be set like beta but for inclusion of 0 and testing its manual
#DECAY = config['params']['decay'] # succeded by LRS

MBATCH = config['params']['mbatch']
DROPOUT =  config['params']['dropout']

LRS_DROP = config['params']['lrs_drop']
LRS_TRESHOLD = config['params']['lrs_treshold']

REPLICATE = config['params']['replicate']

MODELS =  expand("results/{dataset}/{model}_{alpha}_{beta}_{beta2}_{epsilon}_{lrs_drop}_{lrs_treshold}_{dropout}_{mbatch}_{replicate}.csv",
                 dataset=config['input_files']['datasets'],
                 model=config['input_files']['models'],
                 alpha=ALPHA,
                 beta=BETA,
                 beta2=BETA2,
                 epsilon=EPSILON,
                 lrs_drop=LRS_DROP,
                 lrs_treshold=LRS_TRESHOLD,
                 dropout=DROPOUT,
                 mbatch=MBATCH,
                 replicate=REPLICATE)

if not os.path.exists("logs"):
    os.makedirs("logs")

rule all:
    input:
        MODELS

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

#                adam_beta1= config['params']['adam']['beta1']
#                adam_beta2= config['params']['adam']['beta2']
#                adam_epsilon = config['params']['adam']['epsilon']
                #adam
#                config_new.add_section('adam')
#                config_new.set('adam', "BETA1" , adam_beta1)
#                config_new.set('adam', "BETA2" , adam_beta2)
#                config_new.set('adam', "EPSILON" , adam_epsilon)
                config_new.add_section('main')
                config_new.set('main', "SHUFFLE" , config['params']['shuffle'] )
                config_new.set('main', "MODEL_LOAD" , config['params']['model_load'] )
                config_new.set('main', "LRS" , config['params']['lrs'] )
                config_new.set('main', "LRS_EPOCH_DROP" , config['params']['lrs_epoch_drop'] )
                config_new.set('main', "EPOCHS" , config['params']['epochs'])
                config_new.set('main', "MIN_DELTA" , config['params']['min_delta'])
                config_new.set('main', "PATIENCE" , config['params']['patience'])
                config_new.set('main', "DECAY" , config['params']['decay'])
                config_new.write(configfile)

rule makeConfigs:
    input:
        "defaults/config_default.cfg"
    output:
        "configs/config_{alpha}_{beta}_{beta2}_{epsilon}_{lrs_drop}_{lrs_treshold}_{dropout}_{mbatch}_{replicate}.cfg",
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

                config.set('main', "ALPHA" , wildcards.alpha)
                config.set('main', "BETA" , wildcards.beta)
                config.set('main', "BETA2" , wildcards.beta2)
                config.set('main', "EPSILON" , wildcards.epsilon)
                config.set('main', "LRS_DROP" , wildcards.lrs_drop)
                config.set('main', "LRS_TRESHOLD" , wildcards.lrs_treshold)
                config.set('main', "DROPOUT" , wildcards.dropout)
                config.set('main', "MBATCH" , wildcards.mbatch)
                config.set('main', "REPLICATE" , wildcards.replicate)
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
        config="configs/config_{alpha}_{beta}_{beta2}_{epsilon}_{lrs_drop}_{lrs_treshold}_{dropout}_{mbatch}_{replicate}.cfg"
    output:
        results="results/{dataset}/{model}_{alpha}_{beta}_{beta2}_{epsilon}_{lrs_drop}_{lrs_treshold}_{dropout}_{mbatch}_{replicate}.csv"
    params:
        weights="weights/{dataset}/{model}_{alpha}_{beta}_{beta2}_{epsilon}_{lrs_drop}_{lrs_treshold}_{dropout}_{mbatch}_{replicate}"
    log:
        log1="logs/{dataset}/{model}_{alpha}_{beta}_{beta2}_{epsilon}_{lrs_drop}_{lrs_treshold}_{dropout}_{mbatch}_{replicate}.log"
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
