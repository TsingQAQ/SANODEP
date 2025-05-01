#!/bin/bash
#PBS -l select=1:ncpus=2:mem=32gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=36:00:00
#PBS -J 1-9
#PBS -N parallel_gpu_opt_1_9

# 
# """
# All model evaluation, this script will evaluate model's average performance across different number of context trajectories.
# """

# The environment variable CUDA_VISIBLE_DEVICES will be set automatically
# by the job scheduler based on the allocated GPUs. No need to modify this.

# Manually set the PBS_ARRAY_INDEX to 4
# export PBS_ARRAY_INDEX=9

# Change to home directory
cd $HOME/
module load anaconda3/personal
source activate SANODEP

# Navigate to the project directory
cd $HOME/codes/NeuralProcesses/

# Lotka-Volterra
python main.py --config "./exps/cfgs/lotka_voterra/np.py" --workdir "exps/experiments/lotka_voterra/model_comparison/np" --forcast_prob 0.5  --mode opt --train_seed 0  --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/lotka_voterra/nodep.py" --workdir "exps/experiments/lotka_voterra/model_comparison/nodep" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo'  --model "meta_learn"
python main.py --config "./exps/cfgs/lotka_voterra/sanodep.py" --workdir "exps/experiments/lotka_voterra/model_comparison/sanodep" --forcast_prob 0.5  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo'  --model "meta_learn"
python main.py --config "./exps/cfgs/lotka_voterra/sanodep.py" --workdir "exps/experiments/lotka_voterra/model_comparison/sanodep" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo'  --model "meta_learn"
python main.py --config "./exps/cfgs/lotka_voterra/np.py" --workdir "exps/experiments/lotka_voterra/model_comparison/np" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo'  --model "meta_learn"


# Brusselator
python main.py --config "./exps/cfgs/brusselator/np.py" --workdir "exps/experiments/brusselator/np" --forcast_prob 0.5  --mode opt --train_seed 0  --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/brusselator/nodep.py" --workdir "exps/experiments/brusselator/nodep" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/brusselator/sanodep.py" --workdir "exps/experiments/brusselator/sanodep" --forcast_prob 0.5  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/brusselator/sanodep.py" --workdir "exps/experiments/brusselator/sanodep" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/brusselator/np.py" --workdir "exps/experiments/brusselator/np" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"

# Selkov
python main.py --config "./exps/cfgs/selkov/np.py" --workdir "exps/experiments/selkov/np" --forcast_prob 0.5  --mode opt --train_seed 0  --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/selkov/nodep.py" --workdir "exps/experiments/selkov/nodep" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/selkov/sanodep.py" --workdir "exps/experiments/selkov/sanodep" --forcast_prob 0.5  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/selkov/sanodep.py" --workdir "exps/experiments/selkov/sanodep" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/selkov/np.py" --workdir "exps/experiments/selkov/np" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"

# SIR 
python main.py --config "./exps/cfgs/sir_unnormalized/np.py" --workdir "exps/experiments/sir_unnormalized/np" --forcast_prob 0.5  --mode opt --train_seed 0  --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/sir_unnormalized/nodep.py" --workdir "exps/experiments/sir_unnormalized/nodep" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/sir_unnormalized/sanodep.py" --workdir "exps/experiments/sir_unnormalized/sanodep" --forcast_prob 0.5  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/sir_unnormalized/sanodep.py" --workdir "exps/experiments/sir_unnormalized/sanodep" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/sir_unnormalized/np.py" --workdir "exps/experiments/sir_unnormalized/np" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"

# LV3D
python main.py --config "./exps/cfgs/lotka_voterra_3d/np.py" --workdir "exps/experiments/lotka_voterra_3d/np" --forcast_prob 0.5  --mode opt --train_seed 0  --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/lotka_voterra_3d/nodep.py" --workdir "exps/experiments/lotka_voterra_3d/nodep" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/lotka_voterra_3d/sanodep.py" --workdir "exps/experiments/lotka_voterra_3d/sanodep" --forcast_prob 0.5  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/lotka_voterra_3d/sanodep.py" --workdir "exps/experiments/lotka_voterra_3d/sanodep" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/lotka_voterra_3d/np.py" --workdir "exps/experiments/lotka_voterra_3d/np" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"

# SIRD (4D)
python main.py --config "./exps/cfgs/sird/np.py" --workdir "exps/experiments/sird/np" --forcast_prob 0.5  --mode opt --train_seed 0  --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/sird/nodep.py" --workdir "exps/experiments/sird/nodep" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/sird/sanodep.py" --workdir "exps/experiments/sird/sanodep" --forcast_prob 0.5  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/sird/sanodep.py" --workdir "exps/experiments/sird/sanodep" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
python main.py --config "./exps/cfgs/sird/np.py" --workdir "exps/experiments/sird/np" --forcast_prob 0.0  --mode opt --train_seed 0 --opt_seed $PBS_ARRAY_INDEX --opt_type 'mo' --model "meta_learn"
