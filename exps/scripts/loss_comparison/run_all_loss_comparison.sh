# loss training
./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossCondz0ConddsysLoss 0.5 505 0 5
# ./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossCondz0ConddsysLoss 0.0 505 0 5
# ./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossCondz0ConddsysLoss 1.0 505 0 5
./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossCondz0UconddsysLoss 0.5 505 0 5
# ./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossCondz0UconddsysLoss 0.0 505 0 5
# ./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossCondz0UconddsysLoss 1.0 505 0 5
./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossUncondz0ConddsysLoss 0.5 505 0 5
# ./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossUncondz0ConddsysLoss 0.0 505 0 5
# ./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossUncondz0ConddsysLoss 1.0 505 0 5
./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossUncondz0UnconddsysLoss 0.5 505 0 5
# ./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossUncondz0UnconddsysLoss 0.0 505 0 5
# ./loss_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/sanodep/ NeuralODEProcessMFVILossUncondz0UnconddsysLoss 1.0 505 0 5

# ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/ NeuralODEProcessMFVILossCondz0ConddsysLoss 1.0 0.5 0 5
# ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/ NeuralODEProcessMFVILossCondz0ConddsysLoss 0.0 0.5 0 5
# ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/ NeuralODEProcessMFVILossCondz0UconddsysLoss 1.0 0.5 0 5
# ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/ NeuralODEProcessMFVILossCondz0UconddsysLoss 0.0 0.5 0 5
# ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/ NeuralODEProcessMFVILossUncondz0ConddsysLoss 1.0 0.5 0 5
# ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/ NeuralODEProcessMFVILossUncondz0ConddsysLoss 0.0 0.5 0 5
# ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/ NeuralODEProcessMFVILossUncondz0UnconddsysLoss 1.0 0.5 0 5
# ./loss_comparison_eval.sh exps/cfgs/lotka_voterra/sanodep.py exps/experiments/lotka_voterra/loss_comparison/ NeuralODEProcessMFVILossUncondz0UnconddsysLoss 0.0 0.5 0 5

# calculate the mean and std of the results
# python calculate_statistical_model_performance.py --exp_dir exps/experiments/lotka_voterra/loss_comparison/sanodep/ --rseed_range 0 6 --eval_fc_prob 1.0 --fc_prob 0.5