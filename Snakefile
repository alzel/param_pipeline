import re
import os
import glob
import numpy as np
configfile: "config.yaml"

localrules: all

API_key= os.environ.get("COMET_API_KEY")

MODELS =  expand("{experiment}/results/{dataset}/{dataset}_{model}_{replicate_seed}.csv",
                 experiment=config['experiment_name'],
                 dataset=config['input_files']['datasets'],
                 model=config['input_files']['models'],
                 replicate_seed=config['params']['replicate_seed'])
if not os.path.exists("logs"):
    os.makedirs("logs")

rule all:
    input:
        MODELS

# making directoris for params, whose are not automatic
for dataset in config['input_files']['datasets']:
    os.makedirs(os.path.join(config['experiment_name'], "weights", dataset), exist_ok=True)

rule run_model:
    input:
        dataset=lambda wildcards: config["input_files"]['datasets'][wildcards.dataset]["file"],
        config=config["input_files"]["hparam_config"],
        model=lambda wildcards: config["input_files"]["models"][wildcards.model],
    output:
        results="{experiment}/results/{dataset}/{dataset}_{model}_{replicate_seed}.csv"
    params:
        weights="{experiment}/weights/{dataset}/{dataset}_{model}_{replicate_seed}"
    log:
        log1="{experiment}/logs/{dataset}_{model}_{replicate_seed}.log"
    run:
        import sys
        import os
        import socket

        # hacks needed to run different tensoflows depending if node has GPU
        if "kebnekaise" in socket.gethostname():
            prefix = (
                "module load Anaconda3 GCC/7.3.0-2.30 CUDA/9.2.88 OpenMPI/3.1.1 TensorFlow/1.10.0-Python-3.6.6; source activate py36;")
        elif "hebbe" in socket.gethostname():
            prefix = (
                "module load Anaconda3 GCC/6.4.0-2.28 OpenMPI/2.1.2 CUDA/9.1.85 TensorFlow/1.6.0-Python-3.6.4-CUDA-9.1.85; source activate py36;")

            # particular configuration for my MacBook pro
        if "liv003l" in socket.gethostname():
            prefix = ("source activate py36_tensorflow;")

        app = config["input_files"]['app']
        iterations=config["input_files"]['optimizer_iterations']
        proj_name =config["experiment_name"]
        chunks=config["input_files"]['datasets'][wildcards.dataset]["chunks"]
        reverse=config["input_files"]['datasets'][wildcards.dataset]["reverse"]
        cuda_gpu=config["input_files"]['datasets'][wildcards.dataset]["cuda_gpu"]
        prefix = prefix + " CUDA_VISIBLE_DEVICES="+str(cuda_gpu)

        command = f"python {app} --model {input.model} --data {input.dataset} --param_config {input.config} " \
                  f"--output_file {output.results} --model_ckpt_dir {params.weights} --verbose 0 " \
                  f"--CHUNKS {chunks}  --REPLICATE_SEED {wildcards.replicate_seed} " \
                  f"--optimizer_iterations {iterations} --reverse {reverse} 2>&1| tee {log.log1}"

        print(command)
        os.system(prefix + " " + command)
