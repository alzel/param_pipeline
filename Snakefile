import re
import os
import glob
import numpy as np
configfile: "config.yaml"

localrules: all, makeDefault

np.random.seed(123)
###generating parameters
#alpha very important for sure 1
a = config['params']['alpha']['a']
n = config['params']['alpha']['n']
r = np.around(a*np.random.rand(n), 3)
ALPHA = np.unique(np.around(10 ** r, 5))

#beta momentum
#beta = 1 - 10 ^ r
#r[-a, -b] e.g. r E [-3, -1]
a = config['params']['beta']['a']  # mustbe negative
b = config['params']['beta']['b']  # must be
n = config['params']['beta']['n']
r = (a+1)*np.random.rand(n)
BETA = np.unique(np.around(1 - 10 ** (r + b), 5))
MBATCH = config['params']['mbatch']
DROPOUT =  config['params']['dropout']

MODELS =  expand("results/{model}_{alpha}_{beta}_{dropout}_{mbatch}.csv",
                 model=config['input_files']['models'],
                 alpha=ALPHA,
                 beta=BETA,
                 dropout=DROPOUT,
                 mbatch=MBATCH)

# INFILES =  expand("configs/config_{alpha}_{beta}_{dropout}_{mbatch}.cfg",
#                  alpha=ALPHA,
#                  beta=BETA,
#                  dropout=DROPOUT,
#                  mbatch=MBATCH)

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

                adam_beta1= config['params']['adam']['beta1']
                adam_beta2= config['params']['adam']['beta2']
                adam_epsilon = config['params']['adam']['epsilon']
                epochs = config['params']['epochs']
                #adam
                config_new.add_section('adam')
                config_new.set('adam', "BETA1" , adam_beta1)
                config_new.set('adam', "BETA2" , adam_beta2)
                config_new.set('adam', "EPSILON" , adam_epsilon)
                config_new.add_section('main')
                config_new.set('main', "EPOCHS" , epochs)

                config_new.write(configfile)

rule makeConfigs:
    input:
        "defaults/config_default.cfg"
    output:
        "configs/config_{alpha}_{beta}_{dropout}_{mbatch}.cfg",
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
                config.set('main', "DROPOUT" , wildcards.dropout)
                config.set('main', "MBATCH" , wildcards.mbatch)
                config.write(configfile)

#configuration when running on hebbe
import socket
import getpass

#hacks needed to run different tensoflows depending if node has GPU
if "hebbe24-7" in socket.gethostname() or "hebbe24-5" in socket.gethostname():
    if getpass.getuser() == "alezel":
        shell.prefix("module load Anaconda3; module load CUDA/8.0.44; export LD_LIBRARY_PATH=/c3se/users/alezel/Hebbe/bin/cuda/lib64:$LD_LIBRARY_PATH; source activate /c3se/users/alezel/Hebbe/projects/microbes_metagenomics/environments/py36_tensorflow;")
    if getpass.getuser() == "zrimec":
        shell.prefix("module load Anaconda3; module load CUDA/8.0.44; export LD_LIBRARY_PATH=/c3se/users/zrimec/Hebbe/g_nobackup/bin/cuda/lib64:$LD_LIBRARY_PATH; source activate /c3se/users/zrimec/Hebbe/.conda/envs/py36_tensorflow; ")
elif "hebbe" in socket.gethostname():
    if getpass.getuser() == "zrimec":
        shell.prefix("module load Anaconda3; source activate /c3se/users/zrimec/Hebbe/.conda/envs/py36_tensorflow_noGPU")

#particular configuration for my MacBook pro
if "liv003l" in socket.gethostname():
    shell.prefix("source activate tensorflow36")

import os
if not os.path.exists("weights"):
    os.makedirs("weights")

rule run_model:
    input:
        template=config["input_files"]["template"],
        model="models/{model}_input.py",
        config="configs/config_{alpha}_{beta}_{dropout}_{mbatch}.cfg"
    output:
        results="results/{model}_{alpha}_{beta}_{dropout}_{mbatch}.csv"
    params:
        weights="weights/{model}_{alpha}_{beta}_{dropout}_{mbatch}"
    log: "logs/{model}_{alpha}_{beta}_{dropout}_{mbatch}.log"
    shell:
       """
       jupyter nbconvert --to=python {input.template}
       python {input.template} {input.model} {input.config} {params.weights} {output.results} #sys.arg[1] - model name;  #sys.arg[2] - config; sys.arg[3] - trained models; sys.arg[4] - optimization results;
       """

#old rule
# rule run_model:
#     input:
#         model="models/{model}.ipynb",
#         config="configs/config_{alpha}_{beta}_{dropout}_{mbatch}.cfg"
#     output:
#         results="results/{model}_{alpha}_{beta}_{dropout}_{mbatch}.csv",
#     params:
#         weights="weights/{model}_{alpha}_{beta}_{dropout}_{mbatch}"
#     log: "logs/{model}_{alpha}_{beta}_{dropout}_{mbatch}.log"
#     shell:
#         """
#         jupyter nbconvert --to=python models/{wildcards.model}.ipynb
#         python models/{wildcards.model}.py {input.config} {params.weights} {output.results} #sys.arg[1] - config; sys.arg[2] - trained models; sys.arg[3] - optimization results;
#         """
