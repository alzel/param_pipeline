

import re
import os
import glob
import numpy as np
configfile: "config.yaml"

localrules: all
np.random.seed(config['params']['random_seed'])

# number of seeds defines number of replicates, seed given in filenames
N = config['params']['configs_n']
#REPLICATE_SEED = config['params']['replicate_seed']
API_key= os.environ.get("COMET_API_KEY")

MODELS =  expand("results/{dataset}/{dataset}_{model}_{replicate_seed}.csv",
                 dataset=config['input_files']['datasets'],
                 model=config['input_files']['models'],
                 replicate_seed=config['params']['replicate_seed'])

if not os.path.exists("logs"):
    os.makedirs("logs")

rule all:
    input:
        MODELS

# #making directoris for params, whose are not automatic
for dataset in config['input_files']['datasets']:
    os.makedirs(os.path.join("weights", dataset), exist_ok=True)

rule run_model:
    input:
        dataset=lambda wildcards: config["input_files"]['datasets'][wildcards.dataset]["file"],
        config=config["input_files"]["hparam_config"],
        model=lambda wildcards: config["input_files"]["models"][wildcards.model],
    output:
        results="results/{dataset}/{dataset}_{model}_{replicate_seed}.csv"
    params:
        weights="weights/{dataset}/{dataset}_{model}_{replicate_seed}"
    log:
        log1="logs/{dataset}_{model}_{replicate_seed}.log"
    run:
        import socket
        import os
        prefix = ""

        #hacks needed to run different tensoflows depending if node has GPU
        if "hebbe24-7" in socket.gethostname() or "hebbe24-5" in socket.gethostname():
            prefix = ("module load Anaconda3; module load GCC/6.4.0-2.28 OpenMPI/2.1.2; module load TensorFlow/1.6.0-Python-3.6.4-CUDA-9.1.85; source activate python364;")
        elif "hebbe" in socket.gethostname():
            prefix = ("module load Anaconda3; module load GCC/6.4.0-2.28 OpenMPI/2.1.2; module load TensorFlow/1.6.0-Python-3.6.4; source activate python364;")

        #particular configuration for my MacBook pro
        if "liv003l" in socket.gethostname():
            prefix = ("source activate py36_tensorflow;")

        app = config["input_files"]["app"]
        iterations=config["input_files"]['optimizer_iterations']

        suffix=config["input_files"]["experiment_suffix"]
        proj_name = os.path.splitext(os.path.basename(input.dataset))[0] + suffix
        chunks=config["input_files"]['datasets'][wildcards.dataset]["chunks"]
        reverse=config["input_files"]['datasets'][wildcards.dataset]["reverse"]


        command = f"python {input.app} --model {input.model} --data {input.dataset} --param_config {input.config} " \
                  f"--output_file {output.results} --model_ckpt_dir {params.weights} --verbose 0 --project_name {proj_name} " \
                  f"--CHUNKS {chunks} --api_key {API_key} --REPLICATE_SEED {wildcards.replicate_seed} " \
                  f"--optimizer_iterations {iterations} --reverse {reverse} 2>&1| tee {log.log1}"

        print(command)
        os.system(prefix + command)