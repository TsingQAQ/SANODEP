#!/bin/bash
#PBS -l select=1:ncpus=2:mem=10gb
#PBS -l walltime=48:00:00
#PBS -J 0-499
#PBS -N parallel_job_gp_eval

# Set TensorFlow to use CPU only
export CUDA_VISIBLE_DEVICES=""

# Change to home directory
cd $HOME/
module load anaconda3/personal
source activate SANODEP

# Navigate to the project directory
cd $HOME/codes/NeuralProcesses/

eval_dynamic_num=50

# Compute eval_ctx_traj_size and process_id
eval_ctx_traj_size=$(( ($PBS_ARRAY_INDEX) / $eval_dynamic_num ))
process_idx=$(( $PBS_ARRAY_INDEX % eval_dynamic_num ))

echo "eval_ctx_traj_size=$eval_ctx_traj_size, process_idx=$process_idx"

# Run the Python script with explicit argument quoting
# config: ./exps/cfgs/brusselator/gp.py 
# workdir: ./exps/cfgs/experiments/gp/
# eval_forecast_prob: 1.0
# python main.py "$@" --train_seed 0 --model gp --mode eval_gp --eval_ctx_traj_size "$eval_ctx_traj_size" --process_idx "$process_idx" --eval_step_num "1"
python main.py --config "./exps/cfgs/lotka_voterra/gp.py" --workdir "exps/experiments/lotka_voterra/model_comparison/gp" --eval_forcast_prob 0.0 --train_seed 0 --model gp --mode eval_gp --eval_ctx_traj_size "$eval_ctx_traj_size" --process_idx "$process_idx" --eval_step_num 20
python main.py --config "./exps/cfgs/brusselator/gp.py" --workdir "exps/experiments/brusselator/gp" --eval_forcast_prob 0.0 --train_seed 0 --model gp --mode eval_gp --eval_ctx_traj_size "$eval_ctx_traj_size" --process_idx "$process_idx" --eval_step_num 20
python main.py --config "./exps/cfgs/selkov/gp.py" --workdir "exps/experiments/selkov/gp" --eval_forcast_prob 0.0 --train_seed 0 --model gp --mode eval_gp --eval_ctx_traj_size "$eval_ctx_traj_size" --process_idx "$process_idx" --eval_step_num 20
python main.py --config "./exps/cfgs/sir_unnormalized/gp.py" --workdir "exps/experiments/sir_unnormalized/gp" --eval_forcast_prob 0.0 --train_seed 0 --model gp --mode eval_gp --eval_ctx_traj_size "$eval_ctx_traj_size" --process_idx "$process_idx" --eval_step_num 20
python main.py --config "./exps/cfgs/lotka_voterra_3d/gp.py" --workdir "exps/experiments/lotka_voterra_3d/gp" --eval_forcast_prob 0.0 --train_seed 0 --model gp --mode eval_gp --eval_ctx_traj_size "$eval_ctx_traj_size" --process_idx "$process_idx" --eval_step_num 20
python main.py --config "./exps/cfgs/sird/gp.py" --workdir "exps/experiments/sird/gp" --eval_forcast_prob 0.0 --train_seed 0 --model gp --mode eval_gp --eval_ctx_traj_size "$eval_ctx_traj_size" --process_idx "$process_idx" --eval_step_num 20


python main.py --config "./exps/cfgs/lotka_voterra/gp.py" --workdir "exps/experiments/lotka_voterra/model_comparison/gp" --eval_forcast_prob 1.0 --train_seed 0 --model gp --mode eval_gp --eval_ctx_traj_size "$eval_ctx_traj_size" --process_idx "$process_idx" --eval_step_num 20
python main.py --config "./exps/cfgs/brusselator/gp.py" --workdir "exps/experiments/brusselator/gp" --eval_forcast_prob 1.0 --train_seed 0 --model gp --mode eval_gp --eval_ctx_traj_size "$eval_ctx_traj_size" --process_idx "$process_idx" --eval_step_num 20
python main.py --config "./exps/cfgs/selkov/gp.py" --workdir "exps/experiments/selkov/gp" --eval_forcast_prob 1.0 --train_seed 0 --model gp --mode eval_gp --eval_ctx_traj_size "$eval_ctx_traj_size" --process_idx "$process_idx" --eval_step_num 20
python main.py --config "./exps/cfgs/sir_unnormalized/gp.py" --workdir "exps/experiments/sir_unnormalized/gp" --eval_forcast_prob 1.0 --train_seed 0 --model gp --mode eval_gp --eval_ctx_traj_size "$eval_ctx_traj_size" --process_idx "$process_idx" --eval_step_num 20
python main.py --config "./exps/cfgs/lotka_voterra_3d/gp.py" --workdir "exps/experiments/lotka_voterra_3d/gp" --eval_forcast_prob 1.0 --train_seed 0 --model gp --mode eval_gp --eval_ctx_traj_size "$eval_ctx_traj_size" --process_idx "$process_idx" --eval_step_num 20
python main.py --config "./exps/cfgs/sird/gp.py" --workdir "exps/experiments/sird/gp" --eval_forcast_prob 1.0 --train_seed 0 --model gp --mode eval_gp --eval_ctx_traj_size "$eval_ctx_traj_size" --process_idx "$process_idx" --eval_step_num 20