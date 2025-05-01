#!/bin/bash
#PBS -l select=1:ncpus=10:mem=20gb
#PBS -l walltime=48:00:00
#PBS -J 0-9
#PBS -N parallel_cpu_job

# Set TensorFlow to use CPU only
export CUDA_VISIBLE_DEVICES=""

# Change to home directory
cd $HOME/
module load anaconda3/personal
source activate SANODEP

# Navigate to the project directory
cd $HOME/codes/NeuralProcesses/

# Define the commands
python main.py --config 'exps/cfgs/lotka_voterra/gp.py' --workdir 'exps/experiments/lotka_voterra/model_comparison/gp' --mode 'opt' --forcast_prob '0.5' --train_seed 0 --model 'gp' --opt_type 'mo' --opt_seed $PBS_ARRAY_INDEX
python main.py --config 'exps/cfgs/brusselator/gp.py' --workdir 'exps/experiments/brusselator/gp' --mode 'opt' --forcast_prob '0.5' --train_seed 0 --model 'gp' --opt_type 'mo' --opt_seed $PBS_ARRAY_INDEX
python main.py --config 'exps/cfgs/selkov/gp.py' --workdir 'exps/experiments/selkov/gp' --mode 'opt' --forcast_prob '0.5' --train_seed 0 --model 'gp' --opt_type 'mo' --opt_seed $PBS_ARRAY_INDEX
python main.py --config 'exps/cfgs/sir_unnormalized/gp.py' --workdir 'exps/experiments/sir_unnormalized/gp' --mode 'opt' --forcast_prob '0.5' --train_seed 0 --model 'gp' --opt_type 'mo' --opt_seed $PBS_ARRAY_INDEX
python main.py --config 'exps/cfgs/lotka_voterra_3d/gp.py' --workdir 'exps/experiments/lotka_voterra_3d/gp' --mode 'opt' --forcast_prob '0.5' --train_seed 0 --model 'gp' --opt_type 'mo' --opt_seed $PBS_ARRAY_INDEX
python main.py --config 'exps/cfgs/sird/gp.py' --workdir 'exps/experiments/sird/gp' --mode 'opt' --forcast_prob '0.5' --train_seed 0 --model 'gp' --opt_type 'mo' --opt_seed $PBS_ARRAY_INDEX
