# loss training
# ./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossCondz0ConddsysLoss 0.5 0 4
# ./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossCondz0UconddsysLoss 0.5 0 4
# ./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossUncondz0ConddsysLoss 0.5 0 4
# ./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossUncondz0UnconddsysLoss 0.5 0 4

# Model Training comparison
# LV system
$(dirname "$0")/model_comparison_train.sh ./exps/cfgs/lotka_voterra/sanodep_without_init_cond_aug.py ./exps/experiments/lotka_voterra/model_comparison/sanodep_without_init_cond_aug/ 0.5 302 0 4


# PI-SANODEP loss comparison
## loss with clamp only
# clear all the following folders if exists
# rm -rf ../../experiments/lotka_voterra/model_comparison/pi_sanodep_clamp/
# rm -rf ../../experiments/brusselator/pi_sanodep_clamp/
# rm -rf ../../experiments/selkov/pi_sanodep_clamp/
# rm -rf ../../experiments/sir_unnormalized/pi_sanodep_clamp/
# rm -rf ../../experiments/lotka_voterra_3d/pi_sanodep_clamp/
# rm -rf ../../experiments/sird/pi_sanodep_clamp/
# rm -rf ../../experiments/react_net_opt/pi_sanodep_clamp/
# run the training
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/lotka_voterra/pi_sanodep_clamp.py ./exps/experiments/lotka_voterra/model_comparison/pi_sanodep_clamp/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/brusselator/pi_sanodep_clamp.py ./exps/experiments/brusselator/pi_sanodep_clamp/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/selkov/pi_sanodep_clamp.py ./exps/experiments/selkov/pi_sanodep_clamp/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/sir_unnormalized/pi_sanodep_clamp.py ./exps/experiments/sir_unnormalized/pi_sanodep_clamp/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/lotka_voterra_3d/pi_sanodep_clamp.py ./exps/experiments/lotka_voterra_3d/pi_sanodep_clamp/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/sird/pi_sanodep_clamp.py ./exps/experiments/sird/pi_sanodep_clamp/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/react_net_opt/pi_sanodep_clamp.py ./exps/experiments/react_net_opt/pi_sanodep_clamp/ 0.5 302 0 0

## loss with clamp and le regularization
# clear all the following folders if exists
#rm -rf ../../experiments/lotka_voterra/model_comparison/pi_sanodep_clamp_le_reg/
#rm -rf ../../experiments/brusselator/pi_sanodep_clamp_le_reg/
#rm -rf ../../experiments/selkov/pi_sanodep_clamp_le_reg/
#rm -rf ../../experiments/sir_unnormalized/pi_sanodep_clamp_le_reg/
#rm -rf ../../experiments/lotka_voterra_3d/pi_sanodep_clamp_le_reg/
# rm -rf ../../experiments/sird/pi_sanodep_clamp_le_reg/
# rm -rf ../../experiments/react_net_opt/pi_sanodep_clamp_le_reg/
# run the training
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/lotka_voterra/pi_sanodep_clamp_le_reg.py ./exps/experiments/lotka_voterra/model_comparison/pi_sanodep_clamp_le_reg/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/brusselator/pi_sanodep_clamp_le_reg.py ./exps/experiments/brusselator/pi_sanodep_clamp_le_reg/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/selkov/pi_sanodep_clamp_le_reg.py ./exps/experiments/selkov/pi_sanodep_clamp_le_reg/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/sir_unnormalized/pi_sanodep_clamp_le_reg.py ./exps/experiments/sir_unnormalized/pi_sanodep_clamp_le_reg/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/lotka_voterra_3d/pi_sanodep_clamp_le_reg.py ./exps/experiments/lotka_voterra_3d/pi_sanodep_clamp_le_reg/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/sird/pi_sanodep_clamp_le_reg.py ./exps/experiments/sird/pi_sanodep_clamp_le_reg/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/react_net_opt/pi_sanodep_clamp_le_reg.py ./exps/experiments/react_net_opt/pi_sanodep_clamp_le_reg/ 0.5 302 0 0


## loss with clamp and distance penalization
# clear all the following folders if exists
# rm -rf ../../experiments/lotka_voterra/model_comparison/pi_sanodep_clamp_d_penalty/
# rm -rf ../../experiments/brusselator/pi_sanodep_clamp_d_penalty/
# rm -rf ../../experiments/selkov/pi_sanodep_clamp_d_penalty/
# rm -rf ../../experiments/sir_unnormalized/pi_sanodep_clamp_d_penalty/
# rm -rf ../../experiments/lotka_voterra_3d/pi_sanodep_clamp_d_penalty/
# rm -rf ../../experiments/sird/pi_sanodep_clamp_d_penalty/
# rm -rf ../../experiments/react_net_opt/pi_sanodep_clamp_d_penalty/
# run the training
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/lotka_voterra/pi_sanodep_clamp_d_penalty.py ./exps/experiments/lotka_voterra/model_comparison/pi_sanodep_clamp_d_penalty/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/brusselator/pi_sanodep_clamp_d_penalty.py ./exps/experiments/brusselator/pi_sanodep_clamp_d_penalty/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/selkov/pi_sanodep_clamp_d_penalty.py ./exps/experiments/selkov/pi_sanodep_clamp_d_penalty/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/sir_unnormalized/pi_sanodep_clamp_d_penalty.py ./exps/experiments/sir_unnormalized/pi_sanodep_clamp_d_penalty/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/lotka_voterra_3d/pi_sanodep_clamp_d_penalty.py ./exps/experiments/lotka_voterra_3d/pi_sanodep_clamp_d_penalty/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/sird/pi_sanodep_clamp_d_penalty.py ./exps/experiments/sird/pi_sanodep_clamp_d_penalty/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/react_net_opt/pi_sanodep_clamp_d_penalty.py ./exps/experiments/react_net_opt/pi_sanodep_clamp_d_penalty/ 0.5 302 0 0

## loss with clamp and distance penalization and le regularization
# clear all the following folders if exists
# rm -rf ../../experiments/lotka_voterra/model_comparison/pi_sanodep_clamp_le_reg_d_penalty/
# rm -rf ../../experiments/brusselator/pi_sanodep_clamp_le_reg_d_penalty/
# rm -rf ../../experiments/selkov/pi_sanodep_clamp_le_reg_d_penalty/
# rm -rf ../../experiments/sir_unnormalized/pi_sanodep_clamp_le_reg_d_penalty/
# rm -rf ../../experiments/lotka_voterra_3d/pi_sanodep_clamp_le_reg_d_penalty/
# rm -rf ../../experiments/sird/pi_sanodep_clamp_le_reg_d_penalty/
# rm -rf ../../experiments/react_net_opt/pi_sanodep_clamp_le_reg_d_penalty/
# run the training
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/lotka_voterra/pi_sanodep_clamp_le_reg_d_penalty.py ./exps/experiments/lotka_voterra/model_comparison/pi_sanodep_clamp_le_reg_d_penalty/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/brusselator/pi_sanodep_clamp_le_reg_d_penalty.py ./exps/experiments/brusselator/pi_sanodep_clamp_le_reg_d_penalty/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/selkov/pi_sanodep_clamp_le_reg_d_penalty.py ./exps/experiments/selkov/pi_sanodep_clamp_le_reg_d_penalty/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/sir_unnormalized/pi_sanodep_clamp_le_reg_d_penalty.py ./exps/experiments/sir_unnormalized/pi_sanodep_clamp_le_reg_d_penalty/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/lotka_voterra_3d/pi_sanodep_clamp_le_reg_d_penalty.py ./exps/experiments/lotka_voterra_3d/pi_sanodep_clamp_le_reg_d_penalty/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/sird/pi_sanodep_clamp_le_reg_d_penalty.py ./exps/experiments/sird/pi_sanodep_clamp_le_reg_d_penalty/ 0.5 302 0 0
# $(dirname "$0")/model_comparison_train.sh ./exps/cfgs/react_net_opt/pi_sanodep_clamp_le_reg_d_penalty.py ./exps/experiments/react_net_opt/pi_sanodep_clamp_le_reg_d_penalty/ 0.5 302 0 0

# ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/ NeuralODEProcessMFVILossCondz0ConddsysLoss 0.0 0.5 0 4
# ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/ NeuralODEProcessMFVILossCondz0ConddsysLoss 0.0 0.5 0 4
# ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/ NeuralODEProcessMFVILossCondz0UconddsysLoss 1.0 0.5 0 4
# ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/ NeuralODEProcessMFVILossCondz0UconddsysLoss 0.0 0.5 0 4
# ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/ NeuralODEProcessMFVILossUncondz0ConddsysLoss 1.0 0.5 0 4
# ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/ NeuralODEProcessMFVILossUncondz0ConddsysLoss 0.0 0.5 0 4
# ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/ NeuralODEProcessMFVILossUncondz0UnconddsysLoss 1.0 0.5 0 4
# ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/ NeuralODEProcessMFVILossUncondz0UnconddsysLoss 0.0 0.5 0 4

# calculate the mean and std of the results
# python calculate_statistical_model_performance.py --exp_dir exps/experiments/lotka_voterra/loss_comparison/sanodep/ --rseed_range 0 6 --eval_fc_prob 1.0 --fc_prob 0.5