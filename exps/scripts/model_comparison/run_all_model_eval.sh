# run the evaluation
# Usage: 
# 1. ./run_all_model_eval.sh 0.0 
# 2. ./run_all_model_eval.sh 0.5
# 3. ./run_all_model_eval.sh 1.0
# Check if the parameter is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <value>"
  exit 1
fi

# Get the input parameter
value=$1

# LV
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/lotka_voterra/np.py exps/experiments/lotka_voterra/model_comparison/np/ $value 0.5 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/model_comparison/sanodep/ $value 0.5 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/model_comparison/sanodep/ $value 0.0 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/lotka_voterra/nodep.py exps/experiments/lotka_voterra/model_comparison/nodep/ $value 0.0 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/lotka_voterra/np.py exps/experiments/lotka_voterra/model_comparison/np/ $value 0.0 1  0 4

# GP
# Submit the first job and capture its job ID
job_id=$(qsub $(dirname "$0")/model_comparison_gp_hpc.sh ./exps/cfgs/lotka_voterra/gp.py exps/experiments/lotka_voterra/model_comparison/gp/ $value 20)

# Submit the subsequent jobs with dependencies
job_id=$(qsub -W depend=afterok:$job_id $(dirname "$0")/model_comparison_gp_hpc.sh ./exps/cfgs/brusselator/gp.py exps/experiments/brusselator/gp/ $value 20)
job_id=$(qsub -W depend=afterok:$job_id $(dirname "$0")/model_comparison_gp_hpc.sh ./exps/cfgs/selkov/gp.py exps/experiments/selkov/gp/ $value 20)
job_id=$(qsub -W depend=afterok:$job_id $(dirname "$0")/model_comparison_gp_hpc.sh ./exps/cfgs/sir_unnormalized/gp.py exps/experiments/sir_unnormalized/gp/ $value 20)
job_id=$(qsub -W depend=afterok:$job_id $(dirname "$0")/model_comparison_gp_hpc.sh ./exps/cfgs/lotka_voterra_3d/gp.py exps/experiments/lotka_voterra_3d/gp/ $value 20)
job_id=$(qsub -W depend=afterok:$job_id $(dirname "$0")/model_comparison_gp_hpc.sh ./exps/cfgs/sird/gp.py exps/experiments/sird/gp/ $value 20)

# # Brusselator
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/brusselator/np.py exps/experiments/brusselator/np/ $value 0.5 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/brusselator/sanodep.py exps/experiments/brusselator/sanodep/ $value 0.5 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/brusselator/np.py exps/experiments/brusselator/np/ $value 0.0 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/brusselator/sanodep.py exps/experiments/brusselator/sanodep/ $value 0.0 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/brusselator/nodep.py exps/experiments/brusselator/nodep/ $value 0.0 1  0 4
# 
# # Selkov
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/selkov/np.py exps/experiments/selkov/np/ $value 0.5 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/selkov/sanodep.py exps/experiments/selkov/sanodep/ $value 0.5 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/selkov/np.py exps/experiments/selkov/np/ $value 0.0 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/selkov/sanodep.py exps/experiments/selkov/sanodep/ $value 0.0 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/selkov/nodep.py exps/experiments/selkov/nodep/ $value 0.0 1  0 4
# 
# # SIR
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/sir_unnormalized/np.py exps/experiments/sir_unnormalized/np/ $value 0.5 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/sir_unnormalized/sanodep.py exps/experiments/sir_unnormalized/sanodep/ $value 0.5 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/sir_unnormalized/np.py exps/experiments/sir_unnormalized/np/ $value 0.0 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/sir_unnormalized/sanodep.py exps/experiments/sir_unnormalized/sanodep/ $value 0.0 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/sir_unnormalized/nodep.py exps/experiments/sir_unnormalized/nodep/ $value 0.0 1  0 4
# 
# # Lotka-Volterra 3D
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/lotka_voterra_3d/np.py exps/experiments/lotka_voterra_3d/np/ $value 0.5 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/lotka_voterra_3d/sanodep.py exps/experiments/lotka_voterra_3d/sanodep/ $value 0.5 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/lotka_voterra_3d/np.py exps/experiments/lotka_voterra_3d/np/ $value 0.0 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/lotka_voterra_3d/sanodep.py exps/experiments/lotka_voterra_3d/sanodep/ $value 0.0 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/lotka_voterra_3d/nodep.py exps/experiments/lotka_voterra_3d/nodep/ $value 0.0 1  0 4
# 
# # SIRD
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/sird/np.py exps/experiments/sird/np/ $value 0.5 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/sird/sanodep.py exps/experiments/sird/sanodep/ $value 0.5 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/sird/np.py exps/experiments/sird/np/ $value 0.0 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/sird/sanodep.py exps/experiments/sird/sanodep/ $value 0.0 1  0 4
$(dirname "$0")/model_comparison_eval.sh ./exps/cfgs/sird/nodep.py exps/experiments/sird/nodep/ $value 0.0 1  0 4
# 