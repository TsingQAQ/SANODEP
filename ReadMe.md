# Few-Shot Bayesian Optimisation (BO) through System Aware Neural ODE Processes
This repository provides the official implementation of the paper: [System-Aware Neural ODE Processes for Few-Shot Bayesian Optimization](https://arxiv.org/abs/2406.02352) (accepted at [TMLR](https://openreview.net/forum?id=FFnRLvWefK)).

## 🌟 Overview

**TL, DR**: This project proposes **System-Aware Neural ODE Processes (SANODEP)**, a meta-learned model that captures the distribution of dynamical systems, as well as a **BO framework** to enable few-shot BO for 1. initial conditions and 2. measurement time schedules in previously unseen but related systems.

### 🔧 Key Highlights
- Learns an ODE-based prior from a family of dynamical systems
- Supports few-shot Bayesian Optimisation for initial states and time schedules
- Outperforms GP-based models in low-data regimes

### 📊 Visual Comparison
<table>
  <tr>
    <td align="center"><img src="https://github.com/TsingQAQ/NeuralProcesses/blob/tmlr_release/utils/GP.gif?raw=true" width="400"/></td>
    <td align="center"><img src="https://github.com/TsingQAQ/NeuralProcesses/blob/tmlr_release/utils/SANODEP.gif?raw=true" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>GP</b></td>
    <td align="center"><b>SANODEP</b></td>
  </tr>
</table>


## 🛠 Installation
- Install Jax (with GPU support) (tested with Python version `3.10.12`)
- Install this code repository in dev mode, e.g.,  `pip install -e.`

## 🚀 Usage
All training and optimization tasks are launched via main.py with configuration arguments.

All the scripts to run experiments are expected to be stored in `\exps\scripts\`.

### 🔁 Meta-Learning
To train a SANODEP model on Lotka-Volterra, in the folder of `NeuralProcesses/exps/scripts/model_comparison` run:

```
./model_comparison_train.sh exps/cfgs/lotka_voterra/sanodep.py ./exps/experiments/lotka_voterra/model_comparison/sanodep 0.5 301 0 5
```

*There are also pretrained checkpoint available at [google drive](https://drive.google.com/drive/folders/1oeQjYXUnijsjLlYTmg4rOHobKdG2XLWQ?usp=sharing) (please put it at `exps/experiments/`)

### 🔍 To use the meta-learned model for optimization:
To run optimization using the pre-trained model or GP. in the folder of `NeuralProcesses/exps/scripts/optimization`, run:

```
./optimization_comparison.sh exps/cfgs/lotka_voterra/sanodep.py ./exps/experiments/lotka_voterra/model_comparison/sanodep 0.5 0 20 meta_learn
```

## 📝 Notes
1. If you prefer to optimize a subset of the initial condition of the original problem, please specify `config.experimental_design.initial_cond_mapper` to let the algorithm reconstruct the initial condition from the decision variables
2. For now, the multi-objective optimization is optimized based on the raw time space, if 
you have a model build within $t_0$=0 and $t_1$ = 1 with a `config.data.args.time_scaling_coefficient=10`, then the optimization is performed by treating the time objective function varying from [0, 10]. 
3. The current acquisition optimizer uses Scipy `trust-constr` method which is known to be unstable as can have evaluation outside the design space (https://github.com/scipy/scipy/pull/21257), which is fixed in an upcoming unrelease version, it is hence suggested to always install
the latest scipy release version in future when this issue has been fixed. 

## Reproduce the results:
The following command is used to reproduce the results.
```
# Meta Training
model_comparison_train.sh
model_comparison_gp_hpc.sh
		
# Meta Evaluation
model_comparison_eval_gp.sh
model_comparison_eval.sh
		
# Optimisation
run_optimization_gp_hpc.sh
run_optimization_gp_hpc.sh
```
Note if you prefer directly perform few-shot BO on pre-trained model, they are available at [google drive](https://drive.google.com/drive/folders/1oeQjYXUnijsjLlYTmg4rOHobKdG2XLWQ?usp=sharing) (please put it at `exps/experiments/`)


## 📖 Citation
If you find this work or repository helpful, please kindly consider citing our work:
```
@article{qing2024system,
  title={System-Aware Neural ODE Processes for Few-Shot Bayesian Optimization},
  author={Qing, Jixiang and Langdon, Becky D and Lee, Robert M and Shafei, Behrang and van der Wilk, Mark and Tsay, Calvin and Misener, Ruth},
  journal={arXiv preprint arXiv:2406.02352},
  year={2024}
}
```
