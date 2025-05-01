#!/bin/bash
# run this script with 
# 1. optimization_comparison.sh
# 3. ./optimization_comparison.sh model_cfg workdir 0.5 0 20 model_type
#     model_cfg can be 
#         - exps/cfgs/lotka_voterra/sanodep.py
#         - exps/cfgs/lotka_voterra/np.py
#         - exps/cfgs/lotka_voterra/nodep.py
#         - exps/cfgs/lotka_voterra/gp.py
#     workdir can be 
#         - exps/experiments/lotka_voterra/model_comparison/sanodep
#         - exps/experiments/lotka_voterra/model_comparison/np
#         - exps/experiments/lotka_voterra/model_comparison/nodep
#         - exps/experiments/lotka_voterra/model_comparison/gp
#     model_type can be
#         - gp
CONFIG=$1
WORKDIR=$2
FORCAST_PROB=$3
SEED_START=$4
SEED_END=$5
MODEL=$6

for seed_num in $(seq $SEED_START $SEED_END); do
    # Submit the job and capture the job ID, pass additional arguments to the job script
    JOB_ID=$(sbatch ../single_job.sh --workdir $WORKDIR --config $CONFIG --forcast_prob $FORCAST_PROB --mode opt --train_seed 0 --opt_seed $seed_num --model $MODEL --opt_type mo | awk '{print $4}')
    
    # Wait for the job to finish
    while squeue | grep -q "$JOB_ID"; do
        echo "Waiting for job $JOB_ID to complete..."
        sleep 10 # Check every 10 seconds
    done
    
    echo "Job $JOB_ID completed."
done