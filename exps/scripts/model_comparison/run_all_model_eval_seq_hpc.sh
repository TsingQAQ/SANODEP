#!/bin/bash
#PBS -l select=1:ncpus=2:mem=8gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=36:00:00
#PBS -J 0-4
#PBS -N parallel_gpu_job

# All model evaluation, this script will evaluate model's average performance across different number of context trajectories.

# The environment variable CUDA_VISIBLE_DEVICES will be set automatically by the job scheduler based on the allocated GPUs. No need to modify this.

# Change to home directory
cd $HOME/
module load anaconda3/personal
source activate SANODEP

# Navigate to the project directory
cd $HOME/codes/NeuralProcesses/

# Run the Python script with explicit argument quoting
eval_ctx_traj_size_range=(0 1 2 3 4 5 6 7 8 9 10)

# Lotka Voterra
# forecasting only
# for eval_ctx_traj_size in "${eval_ctx_traj_size_range[@]}"; do
#     echo "Running with eval_ctx_traj_size=${eval_ctx_traj_size}"
#     python main.py --config "./exps/cfgs/lotka_voterra/np.py" --workdir "exps/experiments/lotka_voterra/model_comparison/np" --forcast_prob 0.5 --eval_ctx_traj_size ${eval_ctx_traj_size} --eval_forcast_prob 1.0 --mode eval --train_seed $PBS_ARRAY_INDEX
#     python main.py --config "./exps/cfgs/lotka_voterra/sanodep.py" --workdir "exps/experiments/lotka_voterra/model_comparison/sanodep" --forcast_prob 0.5 --eval_ctx_traj_size ${eval_ctx_traj_size} --eval_forcast_prob 1.0 --mode eval --train_seed $PBS_ARRAY_INDEX
#     python main.py --config "./exps/cfgs/lotka_voterra/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/lotka_voterra/model_comparison/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_ctx_traj_size ${eval_ctx_traj_size} --eval_forcast_prob 1.0 --mode eval --train_seed $PBS_ARRAY_INDEX
#     python main.py --config "./exps/cfgs/lotka_voterra/np.py" --workdir "exps/experiments/lotka_voterra/model_comparison/np" --forcast_prob 0.0 --eval_ctx_traj_size ${eval_ctx_traj_size} --eval_forcast_prob 1.0 --mode eval --train_seed $PBS_ARRAY_INDEX
#     python main.py --config "./exps/cfgs/lotka_voterra/nodep.py" --workdir "exps/experiments/lotka_voterra/model_comparison/nodep" --forcast_prob 0.0 --eval_ctx_traj_size ${eval_ctx_traj_size} --eval_forcast_prob 1.0 --mode eval --train_seed $PBS_ARRAY_INDEX
#     python main.py --config "./exps/cfgs/lotka_voterra/sanodep.py" --workdir "exps/experiments/lotka_voterra/model_comparison/sanodep" --forcast_prob 0.0 --eval_ctx_traj_size ${eval_ctx_traj_size} --eval_forcast_prob 1.0 --mode eval --train_seed $PBS_ARRAY_INDEX
# done

# Lotka-Volterra interpolation only
# for eval_ctx_traj_size in "${eval_ctx_traj_size_range[@]}"; do
#     echo "Running interpolation with eval_ctx_traj_size=${eval_ctx_traj_size}"
#     python main.py --config "./exps/cfgs/lotka_voterra/np.py" --workdir "exps/experiments/lotka_voterra/model_comparison/np" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/lotka_voterra/sanodep.py" --workdir "exps/experiments/lotka_voterra/model_comparison/sanodep" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/lotka_voterra/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/lotka_voterra/model_comparison/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/lotka_voterra/np.py" --workdir "exps/experiments/lotka_voterra/model_comparison/np" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/lotka_voterra/nodep.py" --workdir "exps/experiments/lotka_voterra/model_comparison/nodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/lotka_voterra/sanodep.py" --workdir "exps/experiments/lotka_voterra/model_comparison/sanodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
# done

# 
# Brusselator forecasting only
# for eval_ctx_traj_size in "${eval_ctx_traj_size_range[@]}"; do
#     echo "Running forecasting with eval_ctx_traj_size=${eval_ctx_traj_size}"
#     python main.py --config "./exps/cfgs/brusselator/np.py" --workdir "exps/experiments/brusselator/np" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/brusselator/sanodep.py" --workdir "exps/experiments/brusselator/sanodep" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/brusselator/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/brusselator/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/brusselator/np.py" --workdir "exps/experiments/brusselator/np" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/brusselator/nodep.py" --workdir "exps/experiments/brusselator/nodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/brusselator/sanodep.py" --workdir "exps/experiments/brusselator/sanodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
# done

# Brusselator interpolation only
# for eval_ctx_traj_size in "${eval_ctx_traj_size_range[@]}"; do
#     echo "Running interpolation with eval_ctx_traj_size=${eval_ctx_traj_size}"
#     python main.py --config "./exps/cfgs/brusselator/np.py" --workdir "exps/experiments/brusselator/np" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/brusselator/sanodep.py" --workdir "exps/experiments/brusselator/sanodep" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/brusselator/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/brusselator/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/brusselator/np.py" --workdir "exps/experiments/brusselator/np" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/brusselator/nodep.py" --workdir "exps/experiments/brusselator/nodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/brusselator/sanodep.py" --workdir "exps/experiments/brusselator/sanodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
# done

# Selkov forecasting only
# for eval_ctx_traj_size in "${eval_ctx_traj_size_range[@]}"; do
#     echo "Running forecasting with eval_ctx_traj_size=${eval_ctx_traj_size}"
#     python main.py --config "./exps/cfgs/selkov/np.py" --workdir "exps/experiments/selkov/np" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/selkov/sanodep.py" --workdir "exps/experiments/selkov/sanodep" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/selkov/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/selkov/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/selkov/np.py" --workdir "exps/experiments/selkov/np" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/selkov/nodep.py" --workdir "exps/experiments/selkov/nodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/selkov/sanodep.py" --workdir "exps/experiments/selkov/sanodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
# done

# Selkov interpolation only
# for eval_ctx_traj_size in "${eval_ctx_traj_size_range[@]}"; do
#     echo "Running interpolation with eval_ctx_traj_size=${eval_ctx_traj_size}"
#     python main.py --config "./exps/cfgs/selkov/np.py" --workdir "exps/experiments/selkov/np" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/selkov/sanodep.py" --workdir "exps/experiments/selkov/sanodep" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/selkov/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/selkov/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/selkov/np.py" --workdir "exps/experiments/selkov/np" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/selkov/nodep.py" --workdir "exps/experiments/selkov/nodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/selkov/sanodep.py" --workdir "exps/experiments/selkov/sanodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
# done

# SIR forecasting only
# for eval_ctx_traj_size in "${eval_ctx_traj_size_range[@]}"; do
#     echo "Running forecasting with eval_ctx_traj_size=${eval_ctx_traj_size}"
#     python main.py --config "./exps/cfgs/sir_unnormalized/np.py" --workdir "exps/experiments/sir_unnormalized/np" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/sir_unnormalized/sanodep.py" --workdir "exps/experiments/sir_unnormalized/sanodep" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/sir_unnormalized/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/sir_unnormalized/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/sir_unnormalized/np.py" --workdir "exps/experiments/sir_unnormalized/np" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/sir_unnormalized/nodep.py" --workdir "exps/experiments/sir_unnormalized/nodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
#     python main.py --config "./exps/cfgs/sir_unnormalized/sanodep.py" --workdir "exps/experiments/sir_unnormalized/sanodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
# done

# SIR interpolation only
for eval_ctx_traj_size in "${eval_ctx_traj_size_range[@]}"; do
    echo "Running interpolation with eval_ctx_traj_size=${eval_ctx_traj_size}"
    python main.py --config "./exps/cfgs/sir_unnormalized/np.py" --workdir "exps/experiments/sir_unnormalized/np" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/sir_unnormalized/sanodep.py" --workdir "exps/experiments/sir_unnormalized/sanodep" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/sir_unnormalized/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/sir_unnormalized/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/sir_unnormalized/np.py" --workdir "exps/experiments/sir_unnormalized/np" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/sir_unnormalized/nodep.py" --workdir "exps/experiments/sir_unnormalized/nodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/sir_unnormalized/sanodep.py" --workdir "exps/experiments/sir_unnormalized/sanodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
done

# Lotka-Volterra 3D forecasting only
for eval_ctx_traj_size in "${eval_ctx_traj_size_range[@]}"; do
    echo "Running forecasting with eval_ctx_traj_size=${eval_ctx_traj_size}"
    python main.py --config "./exps/cfgs/lotka_voterra_3d/np.py" --workdir "exps/experiments/lotka_voterra_3d/np" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/lotka_voterra_3d/sanodep.py" --workdir "exps/experiments/lotka_voterra_3d/sanodep" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/lotka_voterra_3d/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/lotka_voterra_3d/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/lotka_voterra_3d/np.py" --workdir "exps/experiments/lotka_voterra_3d/np" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/lotka_voterra_3d/nodep.py" --workdir "exps/experiments/lotka_voterra_3d/nodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/lotka_voterra_3d/sanodep.py" --workdir "exps/experiments/lotka_voterra_3d/sanodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
done

# Lotka-Volterra 3D interpolation only
for eval_ctx_traj_size in "${eval_ctx_traj_size_range[@]}"; do
    echo "Running interpolation with eval_ctx_traj_size=${eval_ctx_traj_size}"
    python main.py --config "./exps/cfgs/lotka_voterra_3d/np.py" --workdir "exps/experiments/lotka_voterra_3d/np" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/lotka_voterra_3d/sanodep.py" --workdir "exps/experiments/lotka_voterra_3d/sanodep" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/lotka_voterra_3d/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/lotka_voterra_3d/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/lotka_voterra_3d/np.py" --workdir "exps/experiments/lotka_voterra_3d/np" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/lotka_voterra_3d/nodep.py" --workdir "exps/experiments/lotka_voterra_3d/nodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/lotka_voterra_3d/sanodep.py" --workdir "exps/experiments/lotka_voterra_3d/sanodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
done

# SIRD forecasting only
for eval_ctx_traj_size in "${eval_ctx_traj_size_range[@]}"; do
    echo "Running forecasting with eval_ctx_traj_size=${eval_ctx_traj_size}"
    python main.py --config "./exps/cfgs/sird/np.py" --workdir "exps/experiments/sird/np" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/sird/sanodep.py" --workdir "exps/experiments/sird/sanodep" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/sird/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/sird/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/sird/np.py" --workdir "exps/experiments/sird/np" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/sird/nodep.py" --workdir "exps/experiments/sird/nodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/sird/sanodep.py" --workdir "exps/experiments/sird/sanodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
done

# SIRD interpolation only
for eval_ctx_traj_size in "${eval_ctx_traj_size_range[@]}"; do
    echo "Running interpolation with eval_ctx_traj_size=${eval_ctx_traj_size}"
    python main.py --config "./exps/cfgs/sird/np.py" --workdir "exps/experiments/sird/np" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/sird/sanodep.py" --workdir "exps/experiments/sird/sanodep" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/sird/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/sird/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/sird/np.py" --workdir "exps/experiments/sird/np" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/sird/nodep.py" --workdir "exps/experiments/sird/nodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/sird/sanodep.py" --workdir "exps/experiments/sird/sanodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
done

# Reaction network forecasting only
for eval_ctx_traj_size in "${eval_ctx_traj_size_range[@]}"; do
    echo "Running forecasting with eval_ctx_traj_size=${eval_ctx_traj_size}"
    python main.py --config "./exps/cfgs/react_net_opt/np.py" --workdir "exps/experiments/react_net_opt/np" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/react_net_opt/sanodep.py" --workdir "exps/experiments/react_net_opt/sanodep" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/react_net_opt/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/react_net_opt/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/react_net_opt/np.py" --workdir "exps/experiments/react_net_opt/np" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/react_net_opt/nodep.py" --workdir "exps/experiments/react_net_opt/nodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/react_net_opt/sanodep.py" --workdir "exps/experiments/react_net_opt/sanodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 1.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
done

# Reaction network interpolation only
for eval_ctx_traj_size in "${eval_ctx_traj_size_range[@]}"; do
    echo "Running interpolation with eval_ctx_traj_size=${eval_ctx_traj_size}"
    python main.py --config "./exps/cfgs/react_net_opt/np.py" --workdir "exps/experiments/react_net_opt/np" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/react_net_opt/sanodep.py" --workdir "exps/experiments/react_net_opt/sanodep" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/react_net_opt/sanodep_without_init_cond_encode.py" --workdir "exps/experiments/react_net_opt/sanodep_without_init_cond_encode" --forcast_prob 0.5 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/react_net_opt/np.py" --workdir "exps/experiments/react_net_opt/np" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/react_net_opt/nodep.py" --workdir "exps/experiments/react_net_opt/nodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
    python main.py --config "./exps/cfgs/react_net_opt/sanodep.py" --workdir "exps/experiments/react_net_opt/sanodep" --forcast_prob 0.0 --eval_ctx_traj_size "${eval_ctx_traj_size}" --eval_forcast_prob 0.0 --mode eval --train_seed "$PBS_ARRAY_INDEX"
done
