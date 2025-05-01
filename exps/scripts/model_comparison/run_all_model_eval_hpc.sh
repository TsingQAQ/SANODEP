#!/bin/bash
#PBS -l select=1:ncpus=2:mem=8gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=24:00:00
#PBS -J 0-4
#PBS -N parallel_gpu_job

# 
# """
# All model evaluation, this script will evaluate model's average performance across different number of context trajectories.
# """

# The environment variable CUDA_VISIBLE_DEVICES will be set automatically
# by the job scheduler based on the allocated GPUs. No need to modify this.

# Change to home directory
cd $HOME/
module load anaconda3/personal
source activate SANODEP

# Navigate to the project directory
cd $HOME/codes/NeuralProcesses/


# Run the Python script with explicit argument quoting

# TODO: Add the PI-SANODEP example in 

# Lotka Voterra
# eval_forcast_prob 0.5
# python main.py --config "./exps/cfgs/lotka_voterra/np.py" --workdir "exps/experiments/lotka_voterra/model_comparison/np" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/lotka_voterra/sanodep.py" --workdir "exps/experiments/lotka_voterra/model_comparison/sanodep" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/lotka_voterra/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/lotka_voterra/model_comparison/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/lotka_voterra/nodep.py" --workdir "exps/experiments/lotka_voterra/model_comparison/nodep" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/lotka_voterra/sanodep.py" --workdir "exps/experiments/lotka_voterra/model_comparison/sanodep" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/lotka_voterra/np.py" --workdir "exps/experiments/lotka_voterra/model_comparison/np" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# eval_forcast_prob 0.0, only used for nodep and sanodep for comparison
python main.py --config "./exps/cfgs/lotka_voterra/nodep.py" --workdir "exps/experiments/lotka_voterra/model_comparison/nodep" --forcast_prob 0.0 --eval_forcast_prob 0.0 --mode eval --train_seed $PBS_ARRAY_INDEX
python main.py --config "./exps/cfgs/lotka_voterra/sanodep.py" --workdir "exps/experiments/lotka_voterra/model_comparison/sanodep" --forcast_prob 0.0 --eval_forcast_prob 0.0 --mode eval --train_seed $PBS_ARRAY_INDEX

# Brusselator
# eval_forcast_prob 0.5
# python main.py --config "./exps/cfgs/brusselator/np.py" --workdir "exps/experiments/brusselator/np" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/brusselator/sanodep.py" --workdir "exps/experiments/brusselator/sanodep" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/brusselator/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/brusselator/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/brusselator/nodep.py" --workdir "exps/experiments/brusselator/nodep" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/brusselator/sanodep.py" --workdir "exps/experiments/brusselator/sanodep" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/brusselator/np.py" --workdir "exps/experiments/brusselator/np" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# eval_forcast_prob 0.0, only used for nodep and sanodep for comparison
python main.py --config "./exps/cfgs/brusselator/nodep.py" --workdir "exps/experiments/brusselator/nodep" --forcast_prob 0.0 --eval_forcast_prob 0.0 --mode eval --train_seed $PBS_ARRAY_INDEX
python main.py --config "./exps/cfgs/brusselator/sanodep.py" --workdir "exps/experiments/brusselator/sanodep" --forcast_prob 0.0 --eval_forcast_prob 0.0 --mode eval --train_seed $PBS_ARRAY_INDEX

# Selkov
# eval_forcast_prob 0.5
# python main.py --config "./exps/cfgs/selkov/np.py" --workdir "exps/experiments/selkov/np" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/selkov/sanodep.py" --workdir "exps/experiments/selkov/sanodep" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/selkov/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/selkov/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/selkov/nodep.py" --workdir "exps/experiments/selkov/nodep" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/selkov/sanodep.py" --workdir "exps/experiments/selkov/sanodep" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/selkov/np.py" --workdir "exps/experiments/selkov/np" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# eval_forcast_prob 0.0, only used for nodep and sanodep for comparison
python main.py --config "./exps/cfgs/selkov/nodep.py" --workdir "exps/experiments/selkov/nodep" --forcast_prob 0.0 --eval_forcast_prob 0.0 --mode eval --train_seed $PBS_ARRAY_INDEX
python main.py --config "./exps/cfgs/selkov/sanodep.py" --workdir "exps/experiments/selkov/sanodep" --forcast_prob 0.0 --eval_forcast_prob 0.0 --mode eval --train_seed $PBS_ARRAY_INDEX

# SIR system
# eval_forcast_prob 0.5
# python main.py --config "./exps/cfgs/sir_unnormalized/np.py" --workdir "exps/experiments/sir_unnormalized/np" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/sir_unnormalized/sanodep.py" --workdir "exps/experiments/sir_unnormalized/sanodep" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/sir_unnormalized/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/sir_unnormalized/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/sir_unnormalized/nodep.py" --workdir "exps/experiments/sir_unnormalized/nodep" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/sir_unnormalized/sanodep.py" --workdir "exps/experiments/sir_unnormalized/sanodep" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/sir_unnormalized/np.py" --workdir "exps/experiments/sir_unnormalized/np" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# eval_forcast_prob 0.0, only used for nodep and sanodep for comparison
python main.py --config "./exps/cfgs/sir_unnormalized/nodep.py" --workdir "exps/experiments/sir_unnormalized/nodep" --forcast_prob 0.0 --eval_forcast_prob 0.0 --mode eval --train_seed $PBS_ARRAY_INDEX
python main.py --config "./exps/cfgs/sir_unnormalized/sanodep.py" --workdir "exps/experiments/sir_unnormalized/sanodep" --forcast_prob 0.0 --eval_forcast_prob 0.0 --mode eval --train_seed $PBS_ARRAY_INDEX


# Lotka-Volterra 3D system
# eval_forcast_prob 0.5
# python main.py --config "./exps/cfgs/lotka_voterra_3d/np.py" --workdir "exps/experiments/lotka_voterra_3d/np" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/lotka_voterra_3d/sanodep.py" --workdir "exps/experiments/lotka_voterra_3d/sanodep" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/lotka_voterra_3d/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/lotka_voterra_3d/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/lotka_voterra_3d/nodep.py" --workdir "exps/experiments/lotka_voterra_3d/nodep" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/lotka_voterra_3d/sanodep.py" --workdir "exps/experiments/lotka_voterra_3d/sanodep" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/lotka_voterra_3d/np.py" --workdir "exps/experiments/lotka_voterra_3d/np" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# eval_forcast_prob 0.0, only used for nodep and sanodep for comparison
python main.py --config "./exps/cfgs/lotka_voterra_3d/nodep.py" --workdir "exps/experiments/lotka_voterra_3d/nodep" --forcast_prob 0.0 --eval_forcast_prob 0.0 --mode eval --train_seed $PBS_ARRAY_INDEX
python main.py --config "./exps/cfgs/lotka_voterra_3d/sanodep.py" --workdir "exps/experiments/lotka_voterra_3d/sanodep" --forcast_prob 0.0 --eval_forcast_prob 0.0 --mode eval --train_seed $PBS_ARRAY_INDEX

# SIRD system
# eval_forcast_prob 0.5
# python main.py --config "./exps/cfgs/sird/np.py" --workdir "exps/experiments/sird/np" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/sird/sanodep.py" --workdir "exps/experiments/sird/sanodep" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/sird/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/sird/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/sird/nodep.py" --workdir "exps/experiments/sird/nodep" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/sird/sanodep.py" --workdir "exps/experiments/sird/sanodep" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/sird/np.py" --workdir "exps/experiments/sird/np" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# eval_forcast_prob 0.0, only used for nodep and sanodep for comparison
python main.py --config "./exps/cfgs/sird/nodep.py" --workdir "exps/experiments/sird/nodep" --forcast_prob 0.0 --eval_forcast_prob 0.0 --mode eval --train_seed $PBS_ARRAY_INDEX
python main.py --config "./exps/cfgs/sird/sanodep.py" --workdir "exps/experiments/sird/sanodep" --forcast_prob 0.0 --eval_forcast_prob 0.0 --mode eval --train_seed $PBS_ARRAY_INDEX

# Reaction network
# eval_forcast_prob 0.5
# python main.py --config "./exps/cfgs/react_net_opt/np.py" --workdir "exps/experiments/react_net_opt/np" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/react_net_opt/sanodep.py" --workdir "exps/experiments/react_net_opt/sanodep" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/react_net_opt/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/react_net_opt/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/react_net_opt/nodep.py" --workdir "exps/experiments/react_net_opt/nodep" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/react_net_opt/sanodep.py" --workdir "exps/experiments/react_net_opt/sanodep" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# python main.py --config "./exps/cfgs/react_net_opt/np.py" --workdir "exps/experiments/react_net_opt/np" --forcast_prob 0.0 --eval_forcast_prob 0.5 --mode eval --train_seed $PBS_ARRAY_INDEX 
# eval_forcast_prob 0.0, only used for nodep and sanodep for comparison
python main.py --config "./exps/cfgs/react_net_opt/nodep.py" --workdir "exps/experiments/react_net_opt/nodep" --forcast_prob 0.0 --eval_forcast_prob 0.0 --mode eval --train_seed $PBS_ARRAY_INDEX
python main.py --config "./exps/cfgs/react_net_opt/sanodep.py" --workdir "exps/experiments/react_net_opt/sanodep" --forcast_prob 0.0 --eval_forcast_prob 0.0 --mode eval --train_seed $PBS_ARRAY_INDEX
