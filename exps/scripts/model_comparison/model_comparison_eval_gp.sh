# run it with exps/scripts/model_comparison/model_comparison_gp.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/model_comparison/gp 1.0

# Get command line arguments
CONFIG=$1
WORKDIR=$2
EVAL_FORCAST_PROB=$3
EVAL_STEP_NUM=$4
EVAL_CTX_TRAJ_SIZE_START=$5
EVAL_CTX_TRAJ_SIZE_END=$6

# Run Python script
# python main.py --config $CONFIG --workdir $WORKDIR --mode eval_gp --eval_forcast_prob $EVAL_FORCAST_PROB --train_seed 0 --model meta_learn --opt_type so --opt_seed 0 


for ctx_traj_size in $(seq $EVAL_CTX_TRAJ_SIZE_START $EVAL_CTX_TRAJ_SIZE_END); do
    # Submit the job and capture the job ID, pass additional arguments to the job script
    JOB_ID=$(sbatch ../single_job.sh --config "$CONFIG" --workdir "$WORKDIR" --eval_forcast_prob $EVAL_FORCAST_PROB --train_seed 0 --model gp --mode eval_gp --eval_ctx_traj_size $ctx_traj_size --eval_step_num $EVAL_STEP_NUM | awk '{print $4}')
    # Wait for the job to finish
    while squeue | grep -q "$JOB_ID"; do
        echo "Waiting for job $JOB_ID to complete..."
        sleep 10 # Check every 10 seconds
    done
    
    echo "Job $JOB_ID completed."
done
