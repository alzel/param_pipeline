import re
import os
import glob
import numpy as np
configfile: "config.yaml"

import random
import time


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
        dataset=lambda wildcards: config["input_files"]["datasets"][wildcards.dataset]["file"],
        model=lambda wildcards: config["input_files"]["models"][wildcards.model],
    output:
        results="{experiment}/results/{dataset}/{dataset}_{model}_{replicate_seed}.csv"
    params:
        weights="{experiment}/weights/{dataset}/{dataset}_{model}_{replicate_seed}"
    log:
        log1="{experiment}/logs/{dataset}_{model}_{replicate_seed}.log"
    resources:
        gpu = 49,
        mem_frac = 40
    run:
        import subprocess
        import pandas
        from io import StringIO
        
        def get_CUDA():
            command = "nvidia-smi --query-gpu=gpu_bus_id,pstate,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv,nounits,noheader"
            out_string = subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
            # out_string="""00000000:17:00.0, P0, 65, 13, 16278, 2240, 14038
            # 00000000:65:00.0, P0, 30, 0, 16270, 16104, 166
            
            if out_string:
                df = pandas.read_csv(StringIO(out_string), header=None)
                print(df)
                if df.shape != (1, 1):
                    return df[6].idxmin()
                else:
                    return 0
                    
        one_job = config['params']['one_job']
        file_name = "switch.txt"
        cuda_gpu = 0
        try:
            f = open(file_name)
            first_line = f.readline()
            f.close()
            first_line = int(first_line.strip())
            if one_job:
                cuda_gpu = get_CUDA()
            else:
                if first_line:
                    cuda_gpu = 0
                else:
                    cuda_gpu = 1
        finally:
            f = open(file_name, "w")
            f.write(str(cuda_gpu))
            f.close()

        import os
        import socket
        prefix = ""

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
        hparam_config=config["input_files"]["hparam_config"]

        #cuda_gpu=get_CUDA()
        prefix = prefix + " CUDA_VISIBLE_DEVICES="+str(cuda_gpu)
	print(cuda_gpu)


        command = f"python {app} --model {input.model} --data {input.dataset} --param_config {hparam_config} " \
                  f"--output_file {output.results} --model_ckpt_dir {params.weights} --verbose 0 " \
                  f"--REPLICATE_SEED {wildcards.replicate_seed} " \
                  f"--optimizer_iterations {iterations} 2>&1| tee {log.log1}"
        command = prefix + " " + command
        os.system(prefix + " " + command)
