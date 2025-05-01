#!/bin/bash
#SBATCH --mail-type=ALL # Notification for job done
#SBATCH --mail-user=j.qing@imperial.ac.uk # Your actual email

# Activate your environment
source /vol/bitbucket/jqing/neuralprocess/bin/activate
source /vol/cuda/12.2.0/setup.sh
/usr/bin/nvidia-smi
# Change to your project directory
cd /homes/jqing/codes/NeuralProcesses

# Pass all arguments to the Python script
python main.py "$@"