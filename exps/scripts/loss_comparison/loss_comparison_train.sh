#!/bin/bash
# run this script with 
# 1. chmod +x loss_comparison.sh
# 2. ./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ * 505 0.5 0 5
# * is one of the following (the name in brackets are the corresponding directory name, not used in this script)
#     2.1. NeuralODEProcessMFVILossCondz0ConddsysLoss (cond_init_cond_dynamic)
#     2.2. NeuralODEProcessMFVILossCondz0UconddsysLoss (cond_init_uncond_dynamic)
#     2.3. NeuralODEProcessMFVILossUncondz0ConddsysLoss (uncond_init_cond_dynamic)
#     2.4. NeuralODEProcessMFVILossUncondz0UnconddsysLoss (uncond_init_uncond_dynamic)

CONFIG=$1
WORKDIR=$2
Loss_Form=$3
FORCAST_PROB=$4
TRAIN_EPOCH=$5
SEED_START=$6
SEED_END=$7


for seed_num in $(seq $SEED_START $SEED_END); do
    # Submit the job and capture the job ID, pass additional arguments to the job script
    JOB_ID=$(sbatch ../single_job.sh --config $CONFIG --workdir $WORKDIR --loss $Loss_Form --train_epoch $TRAIN_EPOCH --forcast_prob $FORCAST_PROB --mode train --train_seed $seed_num | awk '{print $4}')
    
    # Wait for the job to finish
    while squeue | grep -q "$JOB_ID"; do
        echo "Waiting for job $JOB_ID to complete..."
        sleep 10 # Check every 10 seconds
    done
    
    echo "Job $JOB_ID completed."
done