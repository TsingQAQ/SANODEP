#!/bin/bash
# run this script with 
# 1. model_comparison_eval.sh
#  CONFIG | WORKDIR | EVAL_FORCAST_PROB | FORCAST_PROB | EVAL_SMP_NUM | EVAL_STEP_NUM | EVAL_CTX_TRAJ_SIZE_START| EVAL_CTX_TRAJ_SIZE_END | SEED_START | SEED_END
# 3. ./model_comparison_eval.sh model_cfg workdir 0.5 0.5 1 100 1 15 0 5
#     model_cfg can be 
#         - exps/cfgs/lotka_voterra/sanodep.py
#         - exps/cfgs/lotka_voterra/np.py
#         - exps/cfgs/lotka_voterra/nodep.py
#     workdir is
#         - exps/experiments/lotka_voterra/model_comparison/
#     eval_forcast_prob can be flexible
#         - 1.0: evaluate the model on forcasting problem
#         - 0.0: evaluate the model on interpolating problem
CONFIG=$1
WORKDIR=$2
EVAL_FORCAST_PROB=$3
FORCAST_PROB=$4
EVAL_SMP_NUM=$5
EVAL_STEP_NUM=$6
EVAL_CTX_TRAJ_SIZE_START=$7
EVAL_CTX_TRAJ_SIZE_END=$8
SEED_START=$9
SEED_END=${10}  # Adjusted to correctly reference the 10th positional parameter

# print the parameters
echo "CONFIG: $CONFIG"
echo "WORKDIR: $WORKDIR"
echo "EVAL_FORCAST_PROB: $EVAL_FORCAST_PROB"
echo "FORCAST_PROB: $FORCAST_PROB"
echo "EVAL_SMP_NUM: $EVAL_SMP_NUM"
echo "EVAL_STEP_NUM: $EVAL_STEP_NUM"
echo "EVAL_CTX_TRAJ_SIZE_START: $EVAL_CTX_TRAJ_SIZE_START"
echo "EVAL_CTX_TRAJ_SIZE_END: $EVAL_CTX_TRAJ_SIZE_END"
echo "SEED_START: $SEED_START"
echo "SEED_END: $SEED_END"


for seed_num in $(seq $SEED_START $SEED_END); do
    for ctx_traj_size in $(seq $EVAL_CTX_TRAJ_SIZE_START $EVAL_CTX_TRAJ_SIZE_END); do
        # Submit the job and capture the job ID, pass additional arguments to the job script
        JOB_ID=$(sbatch ../single_job.sh --config "$CONFIG" --workdir "$WORKDIR" --eval_forcast_prob $EVAL_FORCAST_PROB --eval_model_smp_size $EVAL_SMP_NUM --forcast_prob $FORCAST_PROB --mode eval --train_seed $seed_num --eval_ctx_traj_size $ctx_traj_size --eval_step_num $EVAL_STEP_NUM | awk '{print $4}')
        
        # Wait for the job to finish
        while squeue | grep -q "$JOB_ID"; do
            echo "Waiting for job $JOB_ID to complete..."
            sleep 10 # Check every 10 seconds
        done
        
        echo "Job $JOB_ID completed."
    done
done