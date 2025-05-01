#!/bin/bash
# run this script with 
# 1. chmod +x loss_comparison_eval.sh
# 2. ./loss_comparison_eval.sh model_cfg workdir loss_name eval_forcast_prob 0.5 0 5
#     model_cfg is
#         - exps/cfgs/lotka_voterra/sanodep.py
#     workdir is
#         - exps/experiments/lotka_voterra/loss_comparison/
#     loss_name is one of the following
#         - NeuralODEProcessMFVILossCondz0ConddsysLoss (cond_init_cond_dynamic)
#         - NeuralODEProcessMFVILossCondz0UconddsysLoss (cond_init_uncond_dynamic)
#         - NeuralODEProcessMFVILossUncondz0ConddsysLoss (uncond_init_cond_dynamic)
#         - NeuralODEProcessMFVILossUncondz0UnconddsysLoss (uncond_init_uncond_dynamic)
#     eval_forcast_prob can be flexible
#         - 1.0: evaluate the model on forcasting problem
#         - 0.0: evaluate the model on interpolating problem
# Example: 
#   ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossCondz0ConddsysLoss 1.0 0.5 0 5

CONFIG=$1
WORKDIR=$2
Loss_Form=$3
EVAL_FORCAST_PROB=$4
FORCAST_PROB=$5
SEED_START=$6
SEED_END=$7


for seed_num in $(seq $SEED_START $SEED_END); do
    # Submit the job and capture the job ID, pass additional arguments to the job script
    JOB_ID=$(sbatch ../single_job.sh --workdir $WORKDIR --config $CONFIG --loss $Loss_Form --eval_forcast_prob $EVAL_FORCAST_PROB --forcast_prob $FORCAST_PROB --mode eval --train_seed $seed_num | awk '{print $4}')
    
    # Wait for the job to finish
    while squeue | grep -q "$JOB_ID"; do
        echo "Waiting for job $JOB_ID to complete..."
        sleep 10 # Check every 10 seconds
    done
    
    echo "Job $JOB_ID completed."
done