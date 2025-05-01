#!/bin/bash

CONFIG = $1
WORKDIR = $2
MODE=$3
MODEL=$4
OPT_TYPE=$5
START=$6
END=$7

for i in $(seq $START $END); do
    # Submit the job and capture the job ID, pass additional arguments to the job script
    JOB_ID=$(sbatch ./single_job.sh --config $CONFIG --mode $MODE --model $MODEL --opt_type $OPT_TYPE --opt_seed $i | awk '{print $4}')
    
    # Wait for the job to finish
    while squeue | grep -q "$JOB_ID"; do
        echo "Waiting for job $JOB_ID to complete..."
        sleep 60 # Check every 60 seconds
    done
    
    echo "Job $JOB_ID completed."
done