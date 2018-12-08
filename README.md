# param_pipeline
Optimization of hyperparamers for deep learning models using hyperopt and snakemake

Add your params to params.yml and model
Edit config.yml

To run 4 parallel jobs on 2 gpu cluster:

`nohup snakemake -j 4 --resources gpu=200 mem_frac=160 >> _run_hyperas_e500_i1500.log  2>&1 &`

