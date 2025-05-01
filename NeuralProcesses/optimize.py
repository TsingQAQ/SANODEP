import logging
import os

import jax
import tensorflow as tf
from flax.training import checkpoints, train_state
from jax import numpy as np
from ml_collections import ConfigDict
from tqdm import tqdm

from .models import build
from .train.optimizer import create_optimizer

def gp_based_optimization(
    config: ConfigDict,
    workdir: str,
    opt_type: str,
    opt_seed: int,
    across_traj_plot: bool = False,
    within_traj_plot: bool = False,
):
    """
    To the end, we fall back to use tensorflow based library trieste to perform ordinal Baysian Optimization
    This is a much more well-established library for Bayesian Optimization compared with the GPJax library
    since GPJax does not support non-full cov predictive distribution as well as batch BO yet

    The optimization is main tailored from https://github.com/secondmind-labs/trieste/blob/develop/trieste/bayesian_optimizer.py

    :params config: ConfigDict, the configuration dictionary, where the model checkpoints has been stored
    :params workdir: str, the working directory,
    :params opt_type: str, the optimization type, either single_obj or multi_obj
    """
    import pickle
    from functools import partial

    from jax import random
    from jax.random import uniform
    from trieste.acquisition.multi_objective.pareto import (
        Pareto, get_reference_point)
    from trieste.bayesian_optimizer import OBJECTIVE, Mapping
    from trieste.data import Dataset as trieste_Dataset
    from trieste.space import Box
    from trieste.utils import Timer

    from NeuralProcesses.data.datasets import (NP_Dataset,
                                               SingleTrajectoryDataset)
    from NeuralProcesses.models.gpflow.builder import \
        build_stacked_independent_objectives_model
    from NeuralProcesses.models.utils.transformation import IdentityTransform
    from NeuralProcesses.optim.acquisition.function.function import \
        GreyBoxBatchMonteCarloExpectedImprovement
    from NeuralProcesses.optim.acquisition.function.multi_objective import \
        GreyBoxBatchMonteCarloExpectedHypervolumeImprovement
    from NeuralProcesses.optim.acquisition.multi_objective.utils import \
        sample_pareto_front_from_observer
    from NeuralProcesses.optim.acquisition.rule import \
        MinimumDelayConstrainedBatchEfficientGlobalOptimization
    from NeuralProcesses.optim.utils import log_iteration

    def calc_regret(dataset: trieste_Dataset, global_maximum, time_scaling = 1):
        """
        Calculate the regret
        """
        if opt_type == "so":
            return global_maximum - np.max(
                config.experimental_design.obj_func_form(dataset.observations).numpy()
            )
        else:
            existing_obs = np.concatenate(
                [
                    -config.experimental_design.obj_func_form(
                        dataset.observations
                    ).numpy(),
                    dataset.query_points[..., -1:].numpy() * time_scaling,
                ],
                axis=-1,
            )
            existing_obs_below_ref = existing_obs[
                np.all(existing_obs < config.experimental_design.ref_points, axis=-1)
            ]
            if np.size(existing_obs_below_ref) == 0:
                return global_maximum
            else:
                return global_maximum - Pareto(
                    existing_obs_below_ref
                ).hypervolume_indicator(config.experimental_design.ref_points)

    def plot_gp_bo_contour_plot(
        model,
        observer,
        gp_dataset,
        init_cond_range,
        time_range,
        regrets,
        doe_traj_num,
        initial_idx,
        file_handler,
    ):
        from matplotlib import pyplot as plt
        from trieste.acquisition.multi_objective.pareto import Pareto

        ratio = np.linspace(init_cond_range[0], init_cond_range[1], 100)  # [state_dim]
        time = np.linspace(time_range[0], time_range[1], 1000)
        gp_datasets_input = model._models[0]._model.data[0]

        def observer_with_first_state(times, init_cond, t0, t1):
            state = observer(init_cond, np.atleast_1d(times), t0, t1)
            return np.squeeze(state)

        if opt_type == "so":
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
        else:
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
        # Adjust the position of each subplot to prevent the y-axes from overlapping
        box = axes[0].get_position()
        axes[0].set_position(
            [box.x0 - 0.05, box.y0, box.width * 0.9, box.height]
        )  # Adjust the position and width of each subplot

        # get ground truth trajectory
        traj = jax.vmap(
            jax.vmap(observer_with_first_state, in_axes=(0, None, None, None)),
            in_axes=(None, 0, None, None),
        )(time, ratio, time_range[0], time_range[1])
        # get gp model prediction
        ratio_grid, time_grid = np.meshgrid(np.squeeze(ratio), time)
        inputs = (
            np.stack([ratio_grid, time_grid], axis=-1).astype(np.float64).reshape(-1, 2)
        )  # 2 is hard coded
        mean, _ = model.predict(
            tf.cast(inputs, tf.float64)
        )  # tf.cast(inputs, dtype=tf.float64))
        # gp_pred_mean = mean.numpy().reshape(np.stack([ratio_grid, time_grid], axis=-1).shape)
        gp_pred_mean = mean.numpy().reshape(ratio_grid.shape)
        # Assuming ratio is a numpy array with the same shape as the first axis of traj
        # Create a combined dataset with ratio and traj
        combined = np.hstack(
            (ratio.reshape(-1, 1), traj)
        )  # [initial_cond_num, num_timesteps + 1]
        combined_pred = np.hstack((ratio.reshape(-1, 1), gp_pred_mean.T))
        # combined_gp_jax_pred = np.hstack((ratio.reshape(-1, 1), gp_preds.T))
        # Sort combined by the ratio
        combined_sorted = combined[combined[:, 0].argsort()]
        combined_pred_sorted = combined_pred[combined_pred[:, 0].argsort()]
        # combined_gp_jax_pred_sorted = combined_gp_jax_pred[combined_gp_jax_pred[:,0].argsort()]
        # Separate traj from the sorted combined dataset
        traj_sorted = combined_sorted[:, 1:]
        traj_pred_sorted = combined_pred_sorted[:, 1:]
        # traj_pred_sorted_gpjax = combined_gp_jax_pred_sorted[:, 1:]
        # Create a meshgrid
        X, Y = np.meshgrid(time, combined_sorted[:, 0])
        # get the maximum and minimum for colorbar
        vmin = np.min(np.asarray([traj_pred_sorted.min(), traj_sorted.min()]))
        vmax = np.max(np.asarray([traj_pred_sorted.max(), traj_sorted.max()]))

        # gp predct contour
        contour1 = axes[0].contourf(X, Y, traj_pred_sorted, vmin=vmin, vmax=vmax)
        axes[0].set_title("GP prediction")
        # for _init, traj in zip(init, traj_dicts.values()):
        axes[0].scatter(
            np.squeeze(gp_datasets_input[..., 1].numpy()),
            np.squeeze(gp_datasets_input[..., 0].numpy()),
            s=20,
            color="r",
            label="Real State at different trajectory",
        )

        axes[1].set_title("GP prediction")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Initial Condition")

        # grund truth contour
        contour2 = axes[1].contourf(X, Y, traj_sorted, vmin=vmin, vmax=vmax)

        axes[1].set_title("Ground Truth Contour")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Initial Condition")
        # Adjust the position of each subplot to prevent the y-axes from overlapping
        box = axes[1].get_position()
        axes[1].set_position(
            [box.x0 - 0.05, box.y0, box.width * 0.9, box.height]
        )  # Adjust the position and width of each subplot

        # # Move the right subplot a bit to the right
        box = axes[2].get_position()
        axes[2].set_position([box.x0 - 0.05, box.y0, box.width * 0.9, box.height])
        # contour2 = axes[2].contourf(X, Y, traj_pred_sorted_gpjax, vmin=vmin, vmax=vmax)

        # nodep_mse.append(np.mean(traj_sorted - traj_pred) ** 2)
        # gp_mse.append(np.mean(traj_sorted - gp_preds) ** 2)
        axes[2].plot(
            np.arange(len(regrets)), np.asarray(regrets), label="Simple Regret"
        )
        _ymin, _ymax = axes[2].get_ylim()
        for idx in initial_idx:
            axes[2].vlines(x=idx, ymin=_ymin, ymax=_ymax, color="k", linestyle="--")

        # axes[2].plot(np.arange(1, len(gp_mse)+1, 1), np.asarray(gp_mse), label='GP MSE')
        # axes[2].set_title('model MSE')
        axes[2].set_xlabel("number of observations")
        axes[2].legend()
        axes[2].set_ylabel("Simple Regret")

        if not opt_type == "so":
            axes[3].scatter(
                -gp_dataset.observations[..., 0].numpy(),
                gp_dataset.query_points[..., -1:].numpy(),
                label="Training Point",
            )
            front = Pareto(
                np.concatenate(
                    [
                        -gp_dataset.observations[..., :1].numpy(),
                        gp_dataset.query_points[..., -1:].numpy(),
                    ],
                    axis=-1,
                )
            ).front
            axes[3].scatter(front[:, 0], front[:, 1], label="Pareto Front")
            axes[3].legend()
            axes[3].set_xlabel("Neg Obj")
            axes[3].set_ylabel("Time")
            box = axes[3].get_position()
            axes[3].set_position([box.x0 - 0.05, box.y0, box.width * 0.9, box.height])
            axes[3].set_title("Pareto Front")
        # Add a colorbar
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(contour1, cax=cbar_ax)

        plt.suptitle("Comparison of Model predict vs real Objective Function Contour")
        plt.savefig(f"Lotka_Volterra_gp_pred_{file_handler}.png", dpi=300)

    def plot_gp_bo_contour_plot_paper_usage(
        model,
        observer,
        gp_dataset,
        init_cond_range,
        time_range,
        regrets,
        doe_traj_num,
        initial_idx,
        file_handler,
        initial_cond_mapper,
    ):
        from matplotlib import pyplot as plt
        from trieste.acquisition.multi_objective.pareto import Pareto

        # we assume that we are ploting LV system, as we use the parameterization that x = y =  k (i.e., the ratio)
        # we only use the 1st dim of init_cond to extract the range, as we assume different dim of init_cond has the same range
        ratio = np.linspace(init_cond_range[0], init_cond_range[1], 100)  # [state_dim]
        time = np.linspace(time_range[0], time_range[1], 1000)
        gp_datasets_input = model._models[0]._model.data[0]

        def observer_with_first_state(times, init_cond, t0, t1):
            # this is hard coded for LV system
            state = observer(init_cond, np.atleast_1d(times), t0, t1)[..., 0]
            return np.squeeze(state)

        _, axes = plt.subplots()  # figsize=(3, 2.5)
        # Adjust the position of each subplot to prevent the y-axes from overlapping
        # box = axes.get_position()
        # axes.set_position(
        #     [box.x0 + 0.05, box.y0, box.width * 0.9, box.height]
        # )  # Adjust the position and width of each subplot

        # get ground truth trajectory
        traj = jax.vmap(
            jax.vmap(observer_with_first_state, in_axes=(0, None, None, None)),
            in_axes=(None, 0, None, None),
        )(time, initial_cond_mapper(ratio), time_range[0], time_range[1])
        # get gp model prediction
        ratio_grid, time_grid = np.meshgrid(np.squeeze(ratio), time)
        # change the input here
        inputs = (
            np.concatenate(
                [initial_cond_mapper(ratio_grid[..., None]), time_grid[..., None]],
                axis=-1,
            )
            .astype(np.float64)
            .reshape(-1, 3)
        )  # 2 is hard coded
        mean, _ = model.predict(
            tf.cast(inputs, tf.float64)
        )  # tf.cast(inputs, dtype=tf.float64))
        # gp_pred_mean = mean.numpy().reshape(np.stack([ratio_grid, time_grid], axis=-1).shape)
        gp_pred_mean = mean.numpy()[..., 0].reshape(*ratio_grid.shape)
        # Assuming ratio is a numpy array with the same shape as the first axis of traj
        # Create a combined dataset with ratio and traj
        combined = np.hstack(
            (ratio.reshape(-1, 1), traj)
        )  # [initial_cond_num, num_timesteps + 1]
        combined_pred = np.hstack((ratio.reshape(-1, 1), gp_pred_mean.T))
        # combined_gp_jax_pred = np.hstack((ratio.reshape(-1, 1), gp_preds.T))
        # Sort combined by the ratio
        combined_sorted = combined[combined[:, 0].argsort()]
        combined_pred_sorted = combined_pred[combined_pred[:, 0].argsort()]
        # combined_gp_jax_pred_sorted = combined_gp_jax_pred[combined_gp_jax_pred[:,0].argsort()]
        # Separate traj from the sorted combined dataset
        traj_sorted = combined_sorted[:, 1:]
        traj_pred_sorted = combined_pred_sorted[:, 1:]
        # traj_pred_sorted_gpjax = combined_gp_jax_pred_sorted[:, 1:]
        # Create a meshgrid
        X, Y = np.meshgrid(time, combined_sorted[:, 0])
        # get the maximum and minimum for colorbar
        vmin = np.min(np.asarray([traj_pred_sorted.min(), traj_sorted.min()]))
        vmax = np.max(np.asarray([traj_pred_sorted.max(), traj_sorted.max()]))

        # gp predct contour
        contour1 = axes.contourf(X, Y, traj_pred_sorted, vmin=vmin, vmax=vmax)
        # axes.set_title("GP prediction")
        # for _init, traj in zip(init, traj_dicts.values()):
        axes.scatter(
            np.squeeze(gp_datasets_input[..., -1].numpy())[:10],
            np.squeeze(gp_datasets_input[..., 0].numpy())[:10],
            s=30,
            marker="o",
            color="r",
            facecolors="none",
            label="Real State at different trajectory",
        )
        axes.scatter(
            np.squeeze(gp_datasets_input[..., -1].numpy())[10:],
            np.squeeze(gp_datasets_input[..., 0].numpy())[10:],
            s=30,
            color="r",
            label="Real State at different trajectory",
        )
        axes.tick_params(axis="both", which="major", labelsize=10)
        plt.suptitle("GP prediction", fontsize=20)
        plt.ylabel("Initial Condition", fontsize=14)
        plt.xlabel("Time", fontsize=14)
        plt.tight_layout()
        # Add a colorbar
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # fig.colorbar(contour1, cax=cbar_ax)
        # get current dir
        current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        fig_dir = os.path.join(current_dir, 'exps/experiments/figs/gp_bo')
        try:
            os.makedirs(fig_dir)
        except:
            pass
        with open(os.path.join(fig_dir, f'Lotka_Volterra_gp_pred_{file_handler}.pkl'), 'wb') as f:
            pickle.dump(plt.gcf(), f)
        plt.savefig(os.path.join(fig_dir, f"Lotka_Volterra_gp_pred_{file_handler}.png"), dpi=300)

        # # plot the real function for comparison
        # _, axes = plt.subplots()# figsize=(3, 2.5)
        # # get ground truth trajectory
        # traj = jax.vmap(
        #     jax.vmap(observer_with_first_state, in_axes=(0, None, None, None)),
        #     in_axes=(None, 0, None, None),
        # )(time, ratio, time_range[0], time_range[1])
        # # get gp model prediction
        # ratio_grid, time_grid = np.meshgrid(np.squeeze(ratio), time)
        #
        # # Assuming ratio is a numpy array with the same shape as the first axis of traj
        # # Create a combined dataset with ratio and traj
        # combined = np.hstack(
        #     (ratio.reshape(-1, 1), traj)
        # )  # [initial_cond_num, num_timesteps + 1]
        # # combined_gp_jax_pred = np.hstack((ratio.reshape(-1, 1), gp_preds.T))
        # # Sort combined by the ratio
        # combined_sorted = combined[combined[:, 0].argsort()]
        # # combined_gp_jax_pred_sorted = combined_gp_jax_pred[combined_gp_jax_pred[:,0].argsort()]
        # # Separate traj from the sorted combined dataset
        # traj_sorted = combined_sorted[:, 1:]
        # # traj_pred_sorted_gpjax = combined_gp_jax_pred_sorted[:, 1:]
        # # Create a meshgrid
        # X, Y = np.meshgrid(time, combined_sorted[:, 0])
        # # get the maximum and minimum for colorbar
        # vmin = np.min(np.asarray([traj_pred_sorted.min(), traj_sorted.min()]))
        # vmax = np.max(np.asarray([traj_pred_sorted.max(), traj_sorted.max()]))

        # # gp predct contour
        # contour1 = axes.contourf(X, Y, traj_sorted, vmin=vmin, vmax=vmax)
        # axes.tick_params(axis='both', which='major', labelsize=10)
        # plt.suptitle("Objective Function", fontsize=20)
        # plt.ylabel("Initial Condition", fontsize=14)
        # plt.xlabel("Time", fontsize=14)
        # plt.tight_layout()
        # plt.savefig(f"Lotka_Volterra.png", dpi=400)

    # extract some parameters from config
    opt_rng = random.PRNGKey(opt_seed)
    num_states = config.experimental_design.states_num
    total_num_of_trajs = config.experimental_design.num_traj_iter
    optimize_repeat = config.experimental_design.batch_size_change_times
    maximum_obs_per_traj = config.experimental_design.maximum_obs_per_traj
    init_cond_decision_dim = len(config.experimental_design.x0_lower_bound)

    current_regret = []
    initial_idx = []  # stor all the count where a new trajectory starts
    obs_count = 0

    # design of experiments
    # _, doe_rng = jax.random.split(
    #     opt_rng
    # )  # in gp based opt, the opt_rng is only used to generate the initial ode
    opt_rng, test_problem_rng, doe_rng, _ = random.split(opt_rng, 4)

    observer = partial(
        config.experimental_design.observer, 
        problem_rng=test_problem_rng,
        time_scaling = config.data.args.time_scaling_coefficient
    )
    init_cond_mapper = (
        config.experimental_design.initial_cond_mapper()
        if hasattr(config.experimental_design, "initial_cond_mapper")
        else IdentityTransform()
    )

    # extract global maximum value as a reference of regret
    if (
        not config.experimental_design.fixed_problem
    ):  # the problem can be a generator, we hence also need to calculate corresponding regret
        if opt_type == "so":
            raise NotImplementedError
        elif opt_type == "mo":
            x0_search_space = Box(
                config.experimental_design.x0_lower_bound,
                config.experimental_design.x0_upper_bound,
            )
            t_search_space = Box(
                np.atleast_1d(config.experimental_design.t0),
                np.atleast_1d(config.experimental_design.t1),
            )
            obj = lambda xs, ts: config.experimental_design.obj_func_form(
                observer(
                    xs, ts, config.experimental_design.t0, config.experimental_design.t1
                )
            )
            pf, pf_inputs = sample_pareto_front_from_observer(
                x0_search_space * t_search_space, obj, 50, 
                time_scaling = config.data.args.time_scaling_coefficient, 
                initial_cond_mapper = config.experimental_design.initial_cond_mapper() if hasattr(config.experimental_design, "initial_cond_mapper") else IdentityTransform()
            )
            ref_pts = get_reference_point(pf)
            reference_global_maximum = Pareto(pf).hypervolume_indicator(ref_pts)
        else:
            raise ValueError(f"opt_type: {opt_type} is not supported")
    else:
        reference_global_maximum = config.experimental_design.global_maximum_val

    # rmv log file if it exists
    opt_log_dir = os.path.join(
                workdir,
                "opt",
                f"{opt_type}",
                f"optimization_log_{str(opt_seed)}.csv",
            )
    if os.path.exists(opt_log_dir):
        os.remove(opt_log_dir)

    doe_init_conds = uniform(
        key=doe_rng,  # note that the initial condition dim can be different from the states dim (e.g., in SIR problem)
        shape=(config.experimental_design.initial_traj_num, init_cond_decision_dim),
        minval=config.experimental_design.x0_lower_bound,
        maxval=config.experimental_design.x0_upper_bound,
    )  # [initial_traj_num, state_dim]

    opt_auxilary_infos = {
        "doe_init_conds": doe_init_conds,
        "opt_seed": opt_seed,
        "opt_type": opt_type,
        "reference_global_maximum": reference_global_maximum,
        "acq_init_smp": config.experimental_design.acq_mc_size,
        "acq_opt_par_num": config.experimental_design.acq_opt.acq_opt_parallel_num,
        "acq_opt_max_iter": config.experimental_design.acq_opt.acq_opt_max_iter,
    }

    # create initial dataset
    doe_init_conds = init_cond_mapper(doe_init_conds)
    traj_dicts = {
        f"Traj{traj_idx}": SingleTrajectoryDataset(
            state_dim=init_loc.shape[-1],
            times=config.experimental_design.initial_obs_time,
            observations=observer(
                init_loc,
                config.experimental_design.initial_obs_time,
                config.experimental_design.t0,
                config.experimental_design.t1,
            ),
            initial_cond=init_loc,
        )
        for traj_idx, init_loc in enumerate(doe_init_conds)
    }

    np_datasets = NP_Dataset(
        traj_dicts,
        initial_cond_mapping=(
            config.experimental_design.initial_cond_mapper()
            if hasattr(config.experimental_design, "initial_cond_mapper")
            else IdentityTransform()
        ),
    )
    gp_datasets: trieste_Dataset = np_datasets.formalize_training_data_for_trieste(
        dtype=tf.float64
    )

    # Start Optimization
    logging.info(f"start  {opt_type} based on gp model")

    if opt_type == "so":
        traj_acquisition_builder = GreyBoxBatchMonteCarloExpectedImprovement(
            sample_size=config.experimental_design.acq_mc_size,
            obj_func_form=config.experimental_design.obj_func_form,
            initial_condition_mapping=(
                config.experimental_design.initial_cond_mapper()
                if hasattr(config.experimental_design, "initial_cond_mapper")
                else IdentityTransform()
            ),
        )
    else:
        traj_acquisition_builder = GreyBoxBatchMonteCarloExpectedHypervolumeImprovement(
            sample_size=config.experimental_design.acq_mc_size,
            obj_func_form=config.experimental_design.obj_func_form,
            initial_condition_mapping=(
                config.experimental_design.initial_cond_mapper()
                if hasattr(config.experimental_design, "initial_cond_mapper")
                else IdentityTransform()
            ),
            time_scaling = config.data.args.time_scaling_coefficient,
            # reference_point_spec = config.experimental_design.ref_points,
        )

    traj_acquisition_rule = MinimumDelayConstrainedBatchEfficientGlobalOptimization(
        minimum_delta=config.experimental_design.time_delay,
        builder=traj_acquisition_builder,
        num_query_points=np.arange(
            maximum_obs_per_traj + 1 - optimize_repeat, maximum_obs_per_traj + 1, 1
        ),
        acq_optimizer_initial_smp_num=config.experimental_design.acq_opt.initial_smp_num,
        acq_optimizer_parallel_num=config.experimental_design.acq_opt.acq_opt_parallel_num,
        acq_optimizer_max_iter=config.experimental_design.acq_opt.acq_opt_max_iter,
    )
    models = build_stacked_independent_objectives_model(
        gp_datasets, _num_states=num_states
    )
    current_regret.append(calc_regret(gp_datasets, reference_global_maximum, time_scaling = config.data.args.time_scaling_coefficient))

    existing_initial_trajs = doe_init_conds.shape[0]
    for across_traj_opt_step in np.arange(
        existing_initial_trajs, total_num_of_trajs + existing_initial_trajs, 1
    ):
        pbar = tqdm(
            np.arange(1, maximum_obs_per_traj + 1, 1),
            total=maximum_obs_per_traj,
            initial=0,
            desc=f"trajectory: {across_traj_opt_step}",
        )
        logging.info(f"start optimize {across_traj_opt_step}th initial condition")

        if isinstance(gp_datasets, trieste_Dataset):
            gp_datasets = {OBJECTIVE: gp_datasets}
        if not isinstance(models, Mapping):
            models = {OBJECTIVE: models}

        # Identify the promising initial condition
        # try:
        if across_traj_opt_step == 1:
            # if it is the first time to optimize, we need to train thegp model as well
            for tag, model in models.items():
                gp_dataset = gp_datasets[tag]
                assert gp_dataset is not None
                model.update(gp_dataset)
                model.optimize(gp_dataset)
        if across_traj_plot:
            plot_gp_bo_contour_plot_paper_usage(
                models[OBJECTIVE],
                observer,
                gp_datasets[OBJECTIVE],
                (
                    config.experimental_design.x0_lower_bound,
                    config.experimental_design.x0_upper_bound,
                ),
                (config.experimental_design.t0, config.experimental_design.t1),
                doe_traj_num=config.experimental_design.initial_traj_num,
                regrets=current_regret,
                initial_idx=initial_idx,
                file_handler=f"across_traj_opt_step_{across_traj_opt_step}",
                initial_cond_mapper=(
                    config.experimental_design.initial_cond_mapper()
                    if hasattr(config.experimental_design, "initial_cond_mapper")
                    else IdentityTransform()
                ),
            )

        # acquire the next initial location
        with Timer() as initial_loc_opt_timer:
            points_or_stateful = traj_acquisition_rule.acquire(
                initial_loc_bounds=[
                    config.experimental_design.x0_lower_bound,
                    config.experimental_design.x0_upper_bound,
                ],
                t_bounds=[
                    config.experimental_design.t0,
                    config.experimental_design.t1,
                ],
                model=models,
                datasets=gp_datasets,
            )
        if callable(points_or_stateful):
            acquisition_state, query_points = points_or_stateful(acquisition_state)
        else:
            query_points = points_or_stateful
        identified_optimal_init_loc = tf.reshape(
            query_points[:init_cond_decision_dim], -1
        )
        last_obs_time = config.experimental_design.t0
        observer_output = observer(
            init_cond_mapper(np.asarray(identified_optimal_init_loc.numpy())),
            np.atleast_1d(config.experimental_design.t0),
            config.experimental_design.t0,
            config.experimental_design.t1,
        )
        # add new innitial location in a new starting point
        np_datasets.append_new_traj(
            f"Traj{across_traj_opt_step}",
            SingleTrajectoryDataset(
                state_dim=doe_init_conds.shape[-1],
                times=np.asarray([config.experimental_design.t0]),
                observations=observer_output,
                initial_cond=init_cond_mapper(
                    np.asarray(identified_optimal_init_loc.numpy())
                ),
            ),
        )
        gp_datasets: trieste_Dataset = np_datasets.formalize_training_data_for_trieste(
            dtype=tf.float64
        )

        obs_count += 1
        current_regret.append(calc_regret(gp_datasets, reference_global_maximum, time_scaling = config.data.args.time_scaling_coefficient))
        log_iteration(
            obs_count,
            init_cond_mapper(np.asarray(identified_optimal_init_loc.numpy())),
            last_obs_time,
            observer_output,
            initial_loc_opt_timer.time,
            current_regret[-1],
            log_file_path=os.path.join(
                workdir,
                "opt",
                f"{opt_type}",
                f"optimization_log_{str(opt_seed)}.csv",
            ),
            auxilary_info=opt_auxilary_infos,
        )

        initial_idx.append(obs_count)
        maximum_remainin_obs = maximum_obs_per_traj
        # within trajectory optimization
        for (
            within_traj_opt_step
        ) in pbar:  # np.arange(1, total_num_of_obs_per_traj + 1, 1):
            logging.info(
                f"start optimize {within_traj_opt_step}th obs schedule within {across_traj_opt_step}th trajectory"
            )
            if isinstance(gp_datasets, trieste_Dataset):
                gp_datasets = {OBJECTIVE: gp_datasets}

            for tag, model in models.items():
                gp_dataset = gp_datasets[tag]
                assert gp_dataset is not None
                model.update(gp_dataset)
                model.optimize(gp_dataset)

            if np.size(maximum_remainin_obs) == 0 or maximum_remainin_obs < 1:
                break  # stop trajectory optimization
            if within_traj_plot:
                plot_gp_bo_contour_plot(
                    models[OBJECTIVE],
                    observer,
                    gp_dataset,
                    (
                        config.experimental_design.x0_lower_bound,
                        config.experimental_design.x0_upper_bound,
                    ),
                    (config.experimental_design.t0, config.experimental_design.t1),
                    regrets=current_regret,
                    initial_idx=initial_idx,
                    file_handler=f"{across_traj_opt_step}_{within_traj_opt_step}",
                )

            if within_traj_opt_step == 1:
                query_points = query_points[
                    init_cond_decision_dim:
                ]  # we by pass this optimization step
                within_traj_opt_time = np.inf
            else:
                with Timer() as within_traj_opt_timer:
                    points_or_stateful = traj_acquisition_rule.acquire(
                        initial_loc=identified_optimal_init_loc,
                        t_bounds=[last_obs_time, config.experimental_design.t1],
                        model=models,
                        last_obs_time=last_obs_time,
                        datasets=gp_datasets,
                    )
                within_traj_opt_time = within_traj_opt_timer.time
                if callable(points_or_stateful):
                    acquisition_state, query_points = points_or_stateful(
                        acquisition_state
                    )
                else:
                    query_points = points_or_stateful

            # conduct the time consuming observation
            last_obs_time = np.squeeze(np.asarray(query_points[0].numpy()))
            observer_output = observer(
                init_cond_mapper(np.asarray(identified_optimal_init_loc.numpy())),
                np.atleast_1d(last_obs_time),
                config.experimental_design.t0,
                config.experimental_design.t1,
            )
            # update dataset
            np_datasets.append_obs_within_traj(
                f"Traj{across_traj_opt_step}",
                np.atleast_1d(last_obs_time),
                observer_output,
            )
            gp_datasets: trieste_Dataset = (
                np_datasets.formalize_training_data_for_trieste(dtype=tf.float64)
            )
            obs_count += 1
            current_regret.append(calc_regret(gp_datasets, reference_global_maximum, time_scaling = config.data.args.time_scaling_coefficient))
            log_iteration(
                obs_count,
                init_cond_mapper(np.asarray(identified_optimal_init_loc.numpy())),
                last_obs_time,
                observer_output,
                initial_loc_opt_timer.time,
                current_regret[-1],
                log_file_path=os.path.join(
                    workdir,
                    "opt",
                    f"{opt_type}",
                    f"optimization_log_{str(opt_seed)}.csv",
                ),
                auxilary_info=opt_auxilary_infos,
            )

            # update num of remaining query points within the trajectory
            maximum_remainin_obs = np.floor(
                (config.experimental_design.t1 - last_obs_time)
                / config.experimental_design.time_delay
            ).astype(np.int32)
            # next_num_query_points = np.arange(
            #     maximum_remainin_obs + 1 - optimize_repeat,
            #     maximum_remainin_obs + 1,
            #     1,
            # )
            # next_num_query_points = next_num_query_points[
            #     next_num_query_points >= 1
            # ]  # filter out zero/negative num query points
            # the issue is that maximum_remainin_obs keep decreasing but optimize_repeat remains constant
            next_num_query_points = list(
                range(
                    int(np.ceil((maximum_remainin_obs) / 2)),
                    int(maximum_remainin_obs + 1),
                )
            )
            traj_acquisition_rule.update_num_query_points(next_num_query_points)
            pbar.total = int(within_traj_opt_step + maximum_remainin_obs)
        # reset num of remaining query points within the trajectory
        traj_acquisition_rule.update_num_query_points(
            np.arange(
                config.experimental_design.maximum_obs_per_traj + 1 - optimize_repeat,
                config.experimental_design.maximum_obs_per_traj + 1,
                1,
            )
        )
        with open(
            os.path.join(
                workdir,
                "opt",
                f"{opt_type}",
                f"optimization_seed_{str(opt_seed)}_sampled_traj.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(np_datasets.traj_dicts, f)
        # except Exception as error:  # pylint: disable=broad-except
        #     print(error)


def meta_bayesian_optimization(
    config: ConfigDict,
    workdir: str,
    opt_type: str,
    opt_seed: int,
    across_traj_plot: bool = False,
    within_traj_plot: bool = False,
):
    """
    meta learned model based optimization, currently support
    - SANODEP
    - NeuralProcessAcceptSystemData

    To the end, we fall back to use tensorflow based library trieste to perform ordinal Baysian Optimization
    This is a much more well-established library for Bayesian Optimization compared with the GPJax library
    since GPJax does not support non-full cov predictive distribution as well as batch BO yet

    The optimization is main tailored from https://github.com/secondmind-labs/trieste/blob/develop/trieste/bayesian_optimizer.py
    """
    from functools import partial

    from jax import random

    from NeuralProcesses.models.utils.transformation import IdentityTransform

    from NeuralProcesses.utils.dir_mapping import helper_model_dir_name_mapping

    # --------------- load the flax model ---------------------
    assert isinstance(config.seed, int)  # type check
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    flax_model, initial_params, rng = build.init_model(
        init_rng, config
    )
    # create training state
    if "create_optimizer" in config.training:
        optimizer = config.training["create_optimizer"](config)
    else:
        optimizer = create_optimizer(config)
    aug_training_state = train_state.TrainState.create(
        apply_fn=flax_model.apply, params=initial_params, tx=optimizer
    )
    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Resume training when intermediate checkpoints are detected
    aug_training_state = checkpoints.restore_checkpoint(
        checkpoint_dir, aug_training_state
    )

    # construct a callable model predictor
    @partial(jax.jit, static_argnames="sample_size")
    def flax_model_call(
        context_times,
        context_states,
        context_masks,
        target_times,
        target_initial_cond,
        target_mask,
        initial_time: float,
        end_time: float,
        rng: random.PRNGKey,
        sample_size: int,
    ):
        """
        :params context_times: np.ndarray, [traj_num, num_steps], the time points of the context data
        :params context_states: np.ndarray, [traj_num, num_steps, state_dim], the state of the context data
        :params context_masks: np.ndarray, [traj_num, num_steps], the mask of the context data
        :params target_times: np.ndarray, [traj_num, num_steps], the time points of the target data
        :params target_initial_cond: np.ndarray, [traj_num, state_dim], the initial condition of the target data
        :params target_mask: np.ndarray, the mask of the target data, not actually used here
        :params initial_time: float, the initial time of the target data
        :params end_time: float, the end time of the target data
        :params rng: jax.random.PRNGKey, the random key
        :params sample_size: int, the number of samples
        """
        _traj_size, _num_steps = context_times.shape[0], context_times.shape[1]
        # if the call has been changed, all we need to do is modify here
        # because the test output strutcure of np is currently different than those of sanodep, we
        # have to leverage this ugly hack to make the code work
        if helper_model_dir_name_mapping[config.model.name] == "nodep":
            x_predict_f, x_predict_noise = flax_model.apply(
                aug_training_state.params,
                t_context=context_times,
                x_context=context_states,
                t_target=target_times,
                context_mask=context_masks,
                target_mask=target_mask,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=None,
                training=False,
                solver="Dopri5",
                t0=initial_time,
                t1=end_time,
            )
            return x_predict_f[0], x_predict_noise[0]
        else:
            batch_model_apply = (
                lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: flax_model.apply(
                    aug_training_state.params,
                    t_context=tctx,
                    x_context=x_ctx,
                    t_target=t_tgt,
                    context_mask=mask_ctx,
                    target_mask=mask_tgt,
                    sample_rng=rng,
                    sample_size=sample_size,
                    x_target=x_tgt,
                    training=False,
                    solver="Dopri5",
                    t0=initial_time,
                    t1=end_time,
                    target_initial_cond_mask=np.concatenate(
                        [
                            np.ones(shape=(_traj_size, 1)),
                            np.zeros(shape=(_traj_size, _num_steps - 1)),
                        ],
                        axis=-1,
                    ),
                    ctx_mask_with_new_traj_obs=np.repeat(
                        np.expand_dims(mask_ctx, axis=0), _traj_size, axis=0
                    ),
                    ctx_mask_with_new_traj_target_mask=np.repeat(
                        np.expand_dims(mask_tgt, axis=0), _traj_size, axis=0
                    ),
                )
            )
            if helper_model_dir_name_mapping[config.model.name] == "np":
                # the main difference is that in neural processes, the target time trajectory number should be the same as the context time trajectory number
                x_pred_f, x_pred_noise = batch_model_apply(
                    context_times,
                    context_states,
                    np.repeat(target_times, context_times.shape[0], axis=0),
                    np.repeat(
                        np.repeat(
                            np.expand_dims(target_initial_cond, axis=-2),
                            target_times.shape[-1],
                            axis=-2,
                        ),
                        context_times.shape[0],
                        axis=0,
                    ),
                    context_masks,
                    target_mask,  # target mask is not used in the model
                )

            else:
                x_pred_f, x_pred_noise = batch_model_apply(
                    context_times,
                    context_states,
                    target_times,
                    np.repeat(
                        np.expand_dims(target_initial_cond, axis=-2),
                        target_times.shape[-1],
                        axis=-2,
                    ),
                    context_masks,
                    target_mask,  # target mask is not used in the model
                )
        # note that we only use the first predicted traj, this is a bit hacky but due to the specific design pattern of data formalzaitin
        return x_pred_f[0], x_pred_noise[0]

    # ---------------------------------------------------------
    import pickle

    from jax import random
    from jax.random import uniform
    from trieste.acquisition.multi_objective import get_reference_point
    from trieste.acquisition.multi_objective.pareto import Pareto
    from trieste.bayesian_optimizer import OBJECTIVE, Mapping
    from trieste.data import Dataset as trieste_Dataset
    from trieste.space import Box
    from trieste.utils import Timer

    from NeuralProcesses.data.datasets import (NP_Dataset,
                                               SingleTrajectoryDataset)
    from NeuralProcesses.models.gpflow.models import \
        FlaxDummyGPFlowModelWrapper
    from NeuralProcesses.optim.acquisition.function.function import \
        GreyBoxBatchMonteCarloExpectedImprovementCompatibleWithFlaxModels
    from NeuralProcesses.optim.acquisition.function.multi_objective import \
        GreyBoxBatchMonteCarloExpectedHypervolumeImprovementCompatibleWithFlaxModels
    from NeuralProcesses.optim.acquisition.multi_objective.utils import \
        sample_pareto_front_from_observer
    from NeuralProcesses.optim.acquisition.rule import \
        MinimumDelayConstrainedBatchEfficientGlobalOptimization
    from NeuralProcesses.optim.utils import log_iteration

    def calc_regret(dataset: trieste_Dataset, global_maximum, time_scaling = 1):
        """
        Calculate the regret
        """
        if opt_type == "so":
            return global_maximum - np.max(
                config.experimental_design.obj_func_form(dataset.observations).numpy()
            )
        else:
            existing_obs = np.concatenate(
                [
                    -config.experimental_design.obj_func_form(
                        dataset.observations
                    ).numpy(),
                    dataset.query_points[..., -1:].numpy() * time_scaling,
                ],
                axis=-1,
            )
            existing_obs_below_ref = existing_obs[
                np.all(existing_obs < config.experimental_design.ref_points, axis=-1)
            ]
            if np.size(existing_obs_below_ref) == 0:
                return global_maximum
            else:
                return global_maximum - Pareto(
                    existing_obs_below_ref
                ).hypervolume_indicator(config.experimental_design.ref_points)

    def plot_sanodep_bo_contour_plot(
        model,
        observer,
        init_cond_range,
        time_range,
        regrets,
        dataset,
        initial_idx,
        rng,
        file_handler,
    ):
        @partial(jax.jit, static_argnames=("context_dataset", "sample_size"))
        def model_pred_wrapper(
            target_times, initial_cond, context_dataset, t0, t1, rng, sample_size
        ):
            aug_context_time, aug_context_state, aug_context_mask = (
                context_dataset.formalize_training_data_with_pred_init_cond(
                    np.atleast_2d(initial_cond), np.atleast_2d(t0)
                )
            )
            preds, _ = flax_model_call(
                aug_context_time,
                aug_context_state,
                aug_context_mask,
                np.atleast_2d(target_times),
                np.ones_like(target_times, dtype=np.bool_),
                t0,
                t1,
                rng,
                sample_size,
            )
            # dummy acq defination:
            # return np.squeeze(np.mean(preds, axis=-3)[..., 0])
            return np.squeeze(np.mean(preds, axis=-3))

        from matplotlib import pyplot as plt

        ratio = np.linspace(init_cond_range[0], init_cond_range[1], 100)  # [state_dim]
        time = np.linspace(time_range[0], time_range[1], 1000)

        def observer_with_first_state(times, init_cond, t0, t1):
            state = observer(init_cond, np.atleast_1d(times), t0, t1)
            return np.squeeze(state[:, 0])

        if opt_type == "so":
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
        else:
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))

        # Adjust the position of each subplot to prevent the y-axes from overlapping
        box = axes[0].get_position()
        axes[0].set_position(
            [box.x0 - 0.05, box.y0, box.width * 0.9, box.height]
        )  # Adjust the position and width of each subplot

        # get ground truth trajectory
        traj = jax.vmap(
            jax.vmap(observer_with_first_state, in_axes=(0, None, None, None)),
            in_axes=(None, 0, None, None),
        )(time, ratio, time_range[0], time_range[1])
        initial_conditions = np.concatenate(
            [2 * ratio[..., None], ratio[..., None]], axis=-1
        )
        # TODO: In the future am thinking to merge this traj_pred with model (which is the 1st arg of plot_sanodep_bo_contour_plot)
        traj_pred = jax.vmap(
            jax.vmap(
                model_pred_wrapper, in_axes=(0, None, None, None, None, None, None)
            ),
            in_axes=(None, 0, None, None, None, None, None),
        )(
            time[..., None],
            initial_conditions,
            dataset,
            config.model.t0,
            config.model.t1,
            rng,
            1,
        )

        # Assuming ratio is a numpy array with the same shape as the first axis of traj
        # Create a combined dataset with ratio and traj
        combined = np.hstack(
            (ratio.reshape(-1, 1), traj)
        )  # [initial_cond_num, num_timesteps + 1]
        combined_pred = np.hstack((ratio.reshape(-1, 1), traj_pred))
        # combined_gp_jax_pred = np.hstack((ratio.reshape(-1, 1), gp_preds.T))
        # Sort combined by the ratio
        combined_sorted = combined[combined[:, 0].argsort()]
        combined_pred_sorted = combined_pred[combined_pred[:, 0].argsort()]
        # combined_gp_jax_pred_sorted = combined_gp_jax_pred[combined_gp_jax_pred[:,0].argsort()]
        # Separate traj from the sorted combined dataset
        traj_sorted = combined_sorted[:, 1:]
        traj_pred_sorted = combined_pred_sorted[:, 1:]
        # Create a meshgrid
        X, Y = np.meshgrid(time, combined_sorted[:, 0])
        # get the maximum and minimum for colorbar
        vmin = np.min(np.asarray([traj_pred_sorted.min(), traj_sorted.min()]))
        vmax = np.max(np.asarray([traj_pred_sorted.max(), traj_sorted.max()]))

        # gp predct contour
        contour1 = axes[0].contourf(X, Y, traj_pred_sorted, vmin=vmin, vmax=vmax)
        axes[0].set_title("SANODEP prediction")
        # for _init, traj in zip(init, traj_dicts.values()):
        gp_datasets = dataset.formalize_training_data_for_trieste(dtype=tf.float64)
        gp_datasets_input = gp_datasets.query_points
        axes[0].scatter(
            np.squeeze(gp_datasets_input[..., 1].numpy()),
            np.squeeze(gp_datasets_input[..., 0].numpy()),
            s=20,
            color="r",
            label="Real State at different trajectory",
        )

        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Initial Condition")

        # grund truth contour
        contour2 = axes[1].contourf(X, Y, traj_sorted, vmin=vmin, vmax=vmax)

        axes[1].set_title("Ground Truth Contour")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Initial Condition")
        # Adjust the position of each subplot to prevent the y-axes from overlapping
        box = axes[1].get_position()
        axes[1].set_position(
            [box.x0 - 0.05, box.y0, box.width * 0.9, box.height]
        )  # Adjust the position and width of each subplot

        # # Move the right subplot a bit to the right
        box = axes[2].get_position()
        axes[2].set_position([box.x0 - 0.05, box.y0, box.width * 0.9, box.height])
        # contour2 = axes[2].contourf(X, Y, traj_pred_sorted_gpjax, vmin=vmin, vmax=vmax)

        # nodep_mse.append(np.mean(traj_sorted - traj_pred) ** 2)
        # gp_mse.append(np.mean(traj_sorted - gp_preds) ** 2)
        axes[2].plot(
            np.arange(len(regrets)), np.asarray(regrets), label="Simple Regret"
        )
        _ymin, _ymax = axes[2].get_ylim()
        for idx in initial_idx:
            axes[2].vlines(x=idx, ymin=_ymin, ymax=_ymax, color="k", linestyle="--")

        # axes[2].plot(np.arange(1, len(gp_mse)+1, 1), np.asarray(gp_mse), label='GP MSE')
        # axes[2].set_title('model MSE')
        axes[2].set_xlabel("number of observations")
        axes[2].legend()
        axes[2].set_ylabel("Simple Regret")

        if not opt_type == "so":
            axes[3].scatter(
                -gp_datasets.observations[..., 0].numpy(),
                gp_datasets.query_points[..., -1:].numpy(),
                label="Training Point",
            )
            front = Pareto(
                np.concatenate(
                    [
                        -gp_datasets.observations[..., :1].numpy(),
                        gp_datasets.query_points[..., -1:].numpy(),
                    ],
                    axis=-1,
                )
            ).front
            axes[3].scatter(front[:, 0], front[:, 1], label="Pareto Front")
            axes[3].legend()
            axes[3].set_xlabel("Neg Obj")
            axes[3].set_ylabel("Time")
            box = axes[3].get_position()
            axes[3].set_position([box.x0 - 0.05, box.y0, box.width * 0.9, box.height])
            axes[3].set_title("Pareto Front")
        # Add a colorbar
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(contour1, cax=cbar_ax)

        plt.suptitle("Comparison of Model predict vs real Objective Function Contour")
        current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        fig_dir = os.path.join(current_dir, 'exps/experiments/figs/sanodep_bo')
        try:
            os.makedirs(fig_dir)
        except:
            pass
        plt.savefig(os.path.join(fig_dir, f"Lotka_Volterra_SANODEP_pred_{file_handler}.png"), dpi=300)
        # plt.savefig(f"Lotka_Volterra_SANODEP_pred_{file_handler}.png", dpi=300)

    def plot_sanodep_bo_contour_plot_paper(
        model,
        observer,
        init_cond_range,
        time_range,
        regrets,
        dataset,
        initial_idx,
        rng,
        file_handler,
        initial_cond_mapper=IdentityTransform(),  # note this is a stateless instance hence safe to use as default args
    ):
        helper_model_name_mapping = {
            "NeuralODEProcessAcceptMultiTimeSeriesData": "NODEP",
            "SANODEP": "SANODEP",
            "NeuralProcessAcceptSystemData": "NP",
        }
        from .utils.dir_mapping import helper_model_dir_name_mapping

        # FIXEDME: aug_context_state shape is not changing -> unjit the formalize_training_data_with_pred_init_cond make self.states non static
        # context_dataset.states.shape -> [1, 10, 2] -> aug_context_state.shape = [2, 10, 2]
        # context_dataset.states.shape -> [2, 10, 2] -> aug_context_state.shape = [2, 10, 2] This is wrong context_dataset.states.shape = [2, 10, 2]
        def create_model_pred_wrapper():
            @partial(jax.jit, static_argnames=("context_dataset", "sample_size"))
            def _model_pred_wrapper(
                target_times, initial_cond, context_dataset, t0, t1, rng, sample_size
            ):
                aug_context_time, aug_context_state, aug_context_mask = (
                    context_dataset.formalize_training_data_with_pred_init_cond(
                        np.atleast_2d(initial_cond), np.atleast_2d(t0)
                    )
                )
                preds, _ = flax_model_call(
                    aug_context_time,
                    aug_context_state,
                    aug_context_mask,
                    np.atleast_2d(target_times),
                    np.atleast_2d(initial_cond_mapper(initial_cond)),
                    np.ones_like(target_times, dtype=np.bool_),
                    t0,
                    t1,
                    rng,
                    sample_size,
                )
                # target_initial_cond,
                # target_mask,

                # dummy acq defination:
                # return np.squeeze(np.mean(preds, axis=-3)[..., 0])
                return np.squeeze(np.mean(preds, axis=-3))

            return _model_pred_wrapper

        model_pred_wrapper = create_model_pred_wrapper()
        from matplotlib import pyplot as plt

        ratio = np.linspace(
            init_cond_range[0][0], init_cond_range[1][0], 100
        )  # [state_dim]
        time = np.linspace(time_range[0], time_range[1], 1000)

        def observer_with_first_state(times, init_cond, t0, t1):
            state = observer(init_cond, np.atleast_1d(times), t0, t1)
            return np.squeeze(state[:, 0])

        _, axes = plt.subplots()

        # Adjust the position of each subplot to prevent the y-axes from overlapping
        # box = axes[0].get_position()
        # axes[0].set_position(
        #     [box.x0 - 0.05, box.y0, box.width * 0.9, box.height]
        # )  # Adjust the position and width of each subplot

        # get ground truth trajectory

        initial_conditions = np.concatenate(
            [ratio[..., None], ratio[..., None]], axis=-1
        )
        traj = jax.vmap(
            jax.vmap(observer_with_first_state, in_axes=(0, None, None, None)),
            in_axes=(None, 0, None, None),
        )(
            time, initial_conditions, time_range[0], time_range[1]
        )  # (time, ratio, time_range[0], time_range[1])
        from jax import random

        # debug usage
        # a = model_pred_wrapper(time[..., None][0],
        #     ratio[..., None][0],
        #     dataset,
        #     config.model.t0,
        #     config.model.t1,
        #     random.PRNGKey(0),
        #     32)
        # a = a.mean(-1)
        # # why this is slightly different?
        # b = model.predict(np.atleast_2d(np.concatenate([ratio[..., None][0], time[..., None][0]], axis=-1)))
        # initial_conditions_for_model = ratio[..., None]
        # TODO: In the future am thinking to merge this traj_pred with model (which is the 1st arg of plot_sanodep_bo_contour_plot)
        traj_pred = jax.vmap(
            jax.vmap(
                model_pred_wrapper, in_axes=(0, None, None, None, None, None, None)
            ),
            in_axes=(None, 0, None, None, None, None, None),
        )(
            time[..., None],
            ratio[..., None],
            dataset,
            config.model.t0,
            config.model.t1,
            rng,
            16,
        )[
            ..., 0
        ]

        # Assuming ratio is a numpy array with the same shape as the first axis of traj
        # Create a combined dataset with ratio and traj
        combined = np.hstack(
            (ratio.reshape(-1, 1), traj)
        )  # [initial_cond_num, num_timesteps + 1]
        combined_pred = np.hstack((ratio.reshape(-1, 1), traj_pred))
        # combined_gp_jax_pred = np.hstack((ratio.reshape(-1, 1), gp_preds.T))
        # Sort combined by the ratio
        combined_sorted = combined[combined[:, 0].argsort()]
        combined_pred_sorted = combined_pred[combined_pred[:, 0].argsort()]
        # combined_gp_jax_pred_sorted = combined_gp_jax_pred[combined_gp_jax_pred[:,0].argsort()]
        # Separate traj from the sorted combined dataset
        traj_sorted = combined_sorted[:, 1:]
        traj_pred_sorted = combined_pred_sorted[:, 1:]
        # Create a meshgrid
        X, Y = np.meshgrid(time, combined_sorted[:, 0])
        # get the maximum and minimum for colorbar
        vmin = np.min(np.asarray([traj_pred_sorted.min(), traj_sorted.min()]))
        vmax = np.max(np.asarray([traj_pred_sorted.max(), traj_sorted.max()]))

        # gp predct contour
        contour1 = axes.contourf(X, Y, traj_pred_sorted, vmin=vmin, vmax=vmax)

        # for _init, traj in zip(init, traj_dicts.values()):
        gp_datasets = dataset.formalize_training_data_for_trieste(dtype=tf.float64)
        gp_datasets_input = gp_datasets.query_points
        axes.scatter(
            np.squeeze(gp_datasets_input[..., -1].numpy())[:10],
            np.squeeze(gp_datasets_input[..., 0].numpy())[:10],
            s=30,
            marker="o",
            color="r",
            facecolors="none",
            label="Real State at different trajectory",
        )
        axes.scatter(
            np.squeeze(gp_datasets_input[..., -1].numpy())[10:],
            np.squeeze(gp_datasets_input[..., 0].numpy())[10:],
            s=30,
            color="r",
            label="Real State at different trajectory",
        )
        axes.tick_params(axis="both", which="major", labelsize=10)
        plt.suptitle(
            f"{helper_model_name_mapping[config.model.name]} Prediction", fontsize=20
        )
        plt.ylabel("Initial Condition", fontsize=14)
        plt.xlabel("Time", fontsize=14)
        plt.tight_layout()
        current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        fig_dir = os.path.join(current_dir, F'exps/experiments/figs/{helper_model_name_mapping[config.model.name].lower()}_bo')
        try:
            os.makedirs(fig_dir)
        except:
            pass
        with open(os.path.join(fig_dir, f'Lotka_Volterra_{helper_model_name_mapping[config.model.name]}_pred_{file_handler}.pkl'), 'wb') as f:
            pickle.dump(plt.gcf(), f)
        plt.savefig(os.path.join(fig_dir, f"Lotka_Volterra_{helper_model_name_mapping[config.model.name]}_pred_{file_handler}.png"), dpi=300)
        # plt.savefig(
        #     f"Lotka_Volterra_{helper_model_dir_name_mapping[config.model.name]}_pred_{file_handler}.png",
        #     dpi=300,
        # )

    # extract some parameters from config
    opt_rng = random.PRNGKey(opt_seed)
    opt_rng, test_problem_rng, doe_rng, model_sample_rng = random.split(opt_rng, 4)

    observer = partial(
        config.experimental_design.observer, 
        problem_rng=test_problem_rng,
        time_scaling = config.data.args.time_scaling_coefficient
    )
    num_states = config.experimental_design.states_num
    total_num_of_trajs = config.experimental_design.num_traj_iter
    maximum_obs_per_traj = config.experimental_design.maximum_obs_per_traj
    init_cond_decision_dim = len(config.experimental_design.x0_lower_bound)
    init_cond_mapper = (
        config.experimental_design.initial_cond_mapper()
        if hasattr(config.experimental_design, "initial_cond_mapper")
        else IdentityTransform()
    )
    # extract global maximum value
    if (
        not config.experimental_design.fixed_problem
    ):  # the problem can be a generator, we hence also need to calculate corresponding regret
        if opt_type == "so":
            raise NotImplementedError
        elif opt_type == "mo":
            x0_search_space = Box(
                config.experimental_design.x0_lower_bound,
                config.experimental_design.x0_upper_bound,
            )
            t_search_space = Box(
                np.atleast_1d(config.experimental_design.t0),
                np.atleast_1d(config.experimental_design.t1),
            )
            obj = lambda xs, ts: config.experimental_design.obj_func_form(
                observer(xs, ts)
            )
            pf, pf_inputs = sample_pareto_front_from_observer(
                x0_search_space * t_search_space, obj, 50, 
                time_scaling = config.data.args.time_scaling_coefficient, 
                initial_cond_mapper=config.experimental_design.initial_cond_mapper() if hasattr(config.experimental_design, 'initial_cond_mapper') else IdentityTransform()
            )
            ref_pts = get_reference_point(pf)
            reference_global_maximum = Pareto(pf).hypervolume_indicator(ref_pts)
        else:
            raise ValueError(f"opt_type: {opt_type} is not supported")
    else:
        reference_global_maximum = config.experimental_design.global_maximum_val

    # observer = partial(config.experimental_design.observer, 
    #                    problem_rng=test_problem_rng, 
    #                    t0 = config.experimental_design.t0, 
    #                    t1=config.experimental_design.t1, 
    #                    time_scaling = config.data.args.time_scaling_coefficient)
    # x0_search_space = Box(config.experimental_design.x0_lower_bound, config.experimental_design.x0_upper_bound)
    # t_search_space = Box(np.atleast_1d(config.experimental_design.t0), np.atleast_1d(config.experimental_design.t1))
    # obj = lambda xs, ts: config.experimental_design.obj_func_form(observer(xs, ts))
    # pf, pf_inputs = sample_pareto_front_from_observer(
    #     x0_search_space * t_search_space, 
    #     obj, 50, 
    #     time_scaling = config.data.args.time_scaling_coefficient, 
    #     num_generation=2000, 
    #     initial_cond_mapper=config.experimental_design.initial_cond_mapper() if hasattr(config.experimental_design, 'initial_cond_mapper') else IdentityTransform())
    # ref_pts = get_reference_point(pf)
    # reference_global_maximum = Pareto(pf).hypervolume_indicator(ref_pts)

    # rmv log file if it exists
    opt_log_dir = os.path.join(
                workdir,
                "opt",
                f"{opt_type}",
                f"optimization_log_{str(opt_seed)}.csv",
            )
    if os.path.exists(opt_log_dir):
        os.remove(opt_log_dir)
        
    current_regret = []
    initial_idx = (
        []
    )  # stor all the count where a new trajectory starts, this is only used for plot usage
    obs_count = 0
    optimize_repeat = config.experimental_design.batch_size_change_times

    # design of experiments
    doe_init_conds = uniform(
        key=doe_rng,
        shape=(config.experimental_design.initial_traj_num, init_cond_decision_dim),
        minval=config.experimental_design.x0_lower_bound,
        maxval=config.experimental_design.x0_upper_bound,
    )  # [initial_traj_num, state_dim]

    opt_auxilary_infos = {
        "doe_init_conds": doe_init_conds,
        "opt_seed": opt_seed,
        "opt_type": opt_type,
        "reference_global_maximum": reference_global_maximum,
        "acq_init_smp": config.experimental_design.acq_mc_size,
        "acq_opt_par_num": config.experimental_design.acq_opt.acq_opt_parallel_num,
        "acq_opt_max_iter": config.experimental_design.acq_opt.acq_opt_max_iter,
    }

    # create initial dataset
    # 2024/08/04 temperarlly add for plot usage
    doe_init_conds = init_cond_mapper(
        doe_init_conds
    )  # np.repeat(doe_init_conds, repeats=2, axis=-1)
    traj_dicts = {
        f"Traj{traj_idx}": SingleTrajectoryDataset(
            state_dim=init_loc.shape[-1],
            times=config.experimental_design.initial_obs_time,
            observations=observer(
                init_loc,
                config.experimental_design.initial_obs_time,
                config.experimental_design.t0,
                config.experimental_design.t1,
            ),
            initial_cond=init_loc,
        )
        for traj_idx, init_loc in enumerate(doe_init_conds)
    }

    np_datasets = NP_Dataset(
        traj_dicts,
        initial_cond_mapping=(
            config.experimental_design.initial_cond_mapper()
            if hasattr(config.experimental_design, "initial_cond_mapper")
            else IdentityTransform()
        ),
    )

    # Start Optimization
    logging.info(f"start  {opt_type} based on meta learned model")
    # opt_rng, model_sample_rng = random.split(opt_rng, 2)
    if (
        opt_type == "so"
    ):  # trajectory_aware if enabled will only solve one ode for each initial condition, much more efficient than set it to False
        traj_acquisition_builder = (
            GreyBoxBatchMonteCarloExpectedImprovementCompatibleWithFlaxModels(
                sample_size=config.experimental_design.acq_mc_size,
                sample_rng=model_sample_rng,
                obj_func_form=config.experimental_design.obj_func_form,
                trajectory_aware=config.experimental_design.trajectory_aware,
                initial_condition_mapping=(
                    config.experimental_design.initial_cond_mapper()
                    if hasattr(config.experimental_design, "initial_cond_mapper")
                    else IdentityTransform()
                ),
                time_scaling = config.data.args.time_scaling_coefficient
            )
        )
    else:
        traj_acquisition_builder = GreyBoxBatchMonteCarloExpectedHypervolumeImprovementCompatibleWithFlaxModels(
            sample_size=config.experimental_design.acq_mc_size,
            sample_rng=model_sample_rng,
            obj_func_form=config.experimental_design.obj_func_form,
            trajectory_aware=config.experimental_design.trajectory_aware,
            initial_condition_mapping=(
                config.experimental_design.initial_cond_mapper()
                if hasattr(config.experimental_design, "initial_cond_mapper")
                else IdentityTransform()
            ),
            time_scaling=config.data.args.time_scaling_coefficient,
            # reference_point_spec = config.experimental_design.ref_points,
        )

    traj_acquisition_rule = MinimumDelayConstrainedBatchEfficientGlobalOptimization(
        minimum_delta=config.experimental_design.time_delay,
        builder=traj_acquisition_builder,
        num_query_points=np.arange(
            config.experimental_design.maximum_obs_per_traj + 1 - optimize_repeat,
            config.experimental_design.maximum_obs_per_traj + 1,
            1,
        ),
        acq_optimizer_initial_smp_num=config.experimental_design.acq_opt.initial_smp_num,
        acq_optimizer_parallel_num=config.experimental_design.acq_opt.acq_opt_parallel_num,
        acq_optimizer_max_iter=config.experimental_design.acq_opt.acq_opt_max_iter,
    )
    
    dummy_gp_models = FlaxDummyGPFlowModelWrapper(
        flax_model_call,
        state_dim=num_states,
        t0=config.experimental_design.t0,
        t1=config.experimental_design.t1,
        initial_cond_mapper=(
            config.experimental_design.initial_cond_mapper()
            if hasattr(config.experimental_design, "initial_cond_mapper")
            else IdentityTransform()
        ),
        initial_cond_decision_dim=init_cond_decision_dim,
        trajectory_aware=traj_acquisition_builder.trajectory_aware,
    )
    current_regret.append(
        calc_regret(
            np_datasets.formalize_training_data_for_trieste(dtype=tf.float64),
            reference_global_maximum,
            time_scaling=config.data.args.time_scaling_coefficient,
        )
    )

    existing_initial_trajs = doe_init_conds.shape[0]
    for across_traj_opt_step in np.arange(
        existing_initial_trajs, total_num_of_trajs + existing_initial_trajs, 1
    ):
        pbar = tqdm(
            np.arange(1, maximum_obs_per_traj + 1, 1),
            total=maximum_obs_per_traj,
            initial=0,
            desc=f"trajectory: {across_traj_opt_step}",
        )
        logging.info(f"start optimize {across_traj_opt_step}th initial condition")

        # this has too be kept to maintain compatible interface with trieste
        dummy_gp_datasets = {OBJECTIVE: np_datasets}
        if not isinstance(dummy_gp_models, Mapping):
            dummy_gp_models = {OBJECTIVE: dummy_gp_models}

        # Identify the promising initial condition
        # try:
        if across_traj_opt_step == existing_initial_trajs:
            for tag, model in dummy_gp_models.items():
                dummy_gp_dataset = dummy_gp_datasets[tag]
                assert dummy_gp_dataset is not None
                model.update(dummy_gp_dataset)
                model.optimize(dummy_gp_dataset)

        if across_traj_plot:
            plot_sanodep_bo_contour_plot_paper(
                dummy_gp_models[OBJECTIVE],
                observer,
                (
                    config.experimental_design.x0_lower_bound,
                    config.experimental_design.x0_upper_bound,
                ),
                (config.experimental_design.t0, config.experimental_design.t1),
                dataset=np_datasets,
                regrets=current_regret,
                initial_idx=initial_idx,
                rng=rng,
                file_handler=f"across_traj_opt_step_{across_traj_opt_step}",
                initial_cond_mapper=(
                    config.experimental_design.initial_cond_mapper()
                    if hasattr(config.experimental_design, "initial_cond_mapper")
                    else IdentityTransform()
                ),
            )

        with Timer() as initial_loc_opt_timer:
            points_or_stateful = traj_acquisition_rule.acquire(
                initial_loc_bounds=[
                    config.experimental_design.x0_lower_bound,
                    config.experimental_design.x0_upper_bound,
                ],
                t_bounds=[
                    config.experimental_design.t0,
                    config.experimental_design.t1,
                ],
                model=dummy_gp_models,
                datasets=dummy_gp_datasets,
            )

        if callable(points_or_stateful):
            acquisition_state, query_points = points_or_stateful(acquisition_state)
        else:
            query_points = points_or_stateful
        identified_optimal_init_loc = tf.reshape(
            query_points[:init_cond_decision_dim], -1
        )
        last_obs_time = config.experimental_design.t0
        observer_output = observer(
            init_cond_mapper(np.asarray(identified_optimal_init_loc.numpy())),
            np.atleast_1d(config.experimental_design.t0),
            config.experimental_design.t0,
            config.experimental_design.t1,
        )
        # add new innitial location in a new starting point
        np_datasets.append_new_traj(
            f"Traj{across_traj_opt_step}",
            SingleTrajectoryDataset(
                state_dim=doe_init_conds.shape[-1],
                times=np.asarray([config.experimental_design.t0]),
                observations=observer_output,
                initial_cond=init_cond_mapper(
                    np.asarray(identified_optimal_init_loc.numpy())
                ),
            ),
        )
        _gp_datasets: trieste_Dataset = np_datasets.formalize_training_data_for_trieste(
            dtype=tf.float64
        )

        obs_count += 1
        current_regret.append(calc_regret(_gp_datasets, reference_global_maximum, time_scaling=config.data.args.time_scaling_coefficient,))
        log_iteration(
            obs_count,
            init_cond_mapper(identified_optimal_init_loc.numpy()),
            last_obs_time,
            observer_output,
            initial_loc_opt_timer.time,
            current_regret[-1],
            log_file_path=os.path.join(
                workdir,
                "opt",
                "meta_learn",
                f"{opt_type}",
                f"optimization_log_{str(opt_seed)}.csv",
            ),
            auxilary_info=opt_auxilary_infos,
        )

        initial_idx.append(obs_count)
        maximum_remainin_obs = maximum_obs_per_traj
        # within trajectory optimization
        for (
            within_traj_opt_step
        ) in pbar:  # np.arange(1, total_num_of_obs_per_traj + 1, 1):
            logging.info(
                f"start optimize {within_traj_opt_step}th obs schedule within {across_traj_opt_step}th trajectory"
            )
            dummy_gp_datasets = {OBJECTIVE: np_datasets}
            if not isinstance(dummy_gp_models, Mapping):
                dummy_gp_models = {OBJECTIVE: dummy_gp_models}

            for tag, model in dummy_gp_models.items():
                # Prefer local dataset if available.
                # tags = [tag, LocalizedTag.from_tag(tag).global_tag]
                # _, dataset = get_value_for_tag(filtered_datasets, *tags)
                gp_dataset = dummy_gp_datasets[tag]
                assert gp_dataset is not None
                model.update(gp_dataset)
                model.optimize(gp_dataset)

            if np.size(maximum_remainin_obs) == 0 or maximum_remainin_obs < 1:
                break  # step trajectory optimization

            if within_traj_plot:
                plot_sanodep_bo_contour_plot(
                    dummy_gp_models[OBJECTIVE],
                    observer,
                    (
                        config.experimental_design.x0_lower_bound,
                        config.experimental_design.x0_upper_bound,
                    ),
                    (config.experimental_design.t0, config.experimental_design.t1),
                    dataset=np_datasets,
                    regrets=current_regret,
                    initial_idx=initial_idx,
                    rng=rng,
                    file_handler=f"{across_traj_opt_step}_{within_traj_opt_step}",
                )

            if within_traj_opt_step == 1:
                query_points = query_points[
                    init_cond_decision_dim:
                ]  # we by pass this optimization step
                within_traj_opt_time = np.inf
            else:
                with Timer() as within_traj_opt_timer:
                    points_or_stateful = traj_acquisition_rule.acquire(
                        initial_loc=identified_optimal_init_loc,
                        t_bounds=[last_obs_time, config.experimental_design.t1],
                        model=dummy_gp_models,
                        last_obs_time=last_obs_time,
                        datasets=dummy_gp_datasets,
                    )
                within_traj_opt_time = within_traj_opt_timer.time
                if callable(points_or_stateful):
                    acquisition_state, query_points = points_or_stateful(
                        acquisition_state
                    )
                else:
                    query_points = points_or_stateful

            # conduct the time consuming observation
            last_obs_time = np.squeeze(np.asarray(query_points[0].numpy()))
            observer_output = observer(
                init_cond_mapper(np.asarray(identified_optimal_init_loc.numpy())),
                np.atleast_1d(last_obs_time),
                config.experimental_design.t0,
                config.experimental_design.t1,
            )
            # update dataset, note np_datasets is not updated for flax model predict here but in model.update()
            np_datasets.append_obs_within_traj(
                f"Traj{across_traj_opt_step}",
                np.atleast_1d(last_obs_time),
                observer_output,
            )
            dummy_gp_datasets: trieste_Dataset = (
                np_datasets.formalize_training_data_for_trieste(dtype=tf.float64)
            )

            obs_count += 1
            current_regret.append(
                calc_regret(dummy_gp_datasets, reference_global_maximum, time_scaling=config.data.args.time_scaling_coefficient,)
            )
            log_iteration(
                obs_count,
                init_cond_mapper(identified_optimal_init_loc.numpy()),
                last_obs_time,
                observer_output,
                within_traj_opt_time,
                current_regret[-1],
                log_file_path=os.path.join(
                    workdir,
                    "opt",
                    "meta_learn",
                    f"{opt_type}",
                    f"optimization_log_{str(opt_seed)}.csv",
                ),
            )

            # update num of remaining query points within the trajectory
            maximum_remainin_obs = np.floor(
                (config.experimental_design.t1 - last_obs_time)
                / config.experimental_design.time_delay
            )
            # the issue is that maximum_remainin_obs keep decreasing but optimize_repeat remains constant
            next_num_query_points = list(
                range(
                    int(np.ceil((maximum_remainin_obs) / 2)),
                    int(maximum_remainin_obs + 1),
                )
            )
            traj_acquisition_rule.update_num_query_points(next_num_query_points)
            pbar.total = int(within_traj_opt_step + maximum_remainin_obs)
        # reset num of remaining query points within the trajectory
        traj_acquisition_rule.update_num_query_points(
            np.arange(
                config.experimental_design.maximum_obs_per_traj + 1 - optimize_repeat,
                config.experimental_design.maximum_obs_per_traj + 1,
                1,
            )
        )
        with open(
            os.path.join(
                workdir,
                "opt",
                "meta_learn",
                f"{opt_type}",
                f"optimization_seed_{str(opt_seed)}_sampled_traj.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(np_datasets.traj_dicts, f)
        # except Exception as error:  # pylint: disable=broad-except
        #     print(error)


# TODO: This will be used as a verification to showcase that our framework works
def random_optimization(
    config: ConfigDict, workdir: str, opt_type: str, opt_seed: int, plot: bool = False
):
    """
    To the end, we fall back to use tensorflow based library trieste to perform ordinal Baysian Optimization
    This is a much more well-established library for Bayesian Optimization compared with the GPJax library
    since GPJax does not support non-full cov predictive distribution as well as batch BO yet

    The optimization is main tailored from https://github.com/secondmind-labs/trieste/blob/develop/trieste/bayesian_optimizer.py

    :params config: ConfigDict, the configuration dictionary, where the model checkpoints has been stored
    :params workdir: str, the working directory,
    :params opt_type: str, the optimization type, either single_obj or multi_obj
    """
    import pickle
    from functools import partial

    import gpflow
    import tensorflow_probability as tfp
    from jax import random
    from jax.random import uniform
    from trieste.acquisition.multi_objective.pareto import (
        Pareto, get_reference_point)
    from trieste.bayesian_optimizer import OBJECTIVE, Mapping
    from trieste.data import Dataset as trieste_Dataset
    from trieste.models.gpflow import GaussianProcessRegression
    from trieste.models.interfaces import \
        TrainablePredictJointReparamModelStack
    from trieste.space import Box
    from trieste.utils import Timer

    from NeuralProcesses.data.datasets import (NP_Dataset,
                                               SingleTrajectoryDataset)
    from NeuralProcesses.optim.acquisition.function.function import \
        GreyBoxBatchMonteCarloExpectedImprovement
    from NeuralProcesses.optim.acquisition.function.multi_objective import \
        GreyBoxBatchMonteCarloExpectedHypervolumeImprovement
    from NeuralProcesses.optim.acquisition.multi_objective.utils import \
        sample_pareto_front_from_observer
    from NeuralProcesses.optim.acquisition.rule import \
        MinimumDelayConstrainedBatchEfficientGlobalOptimization
    from NeuralProcesses.optim.utils import log_iteration
    from NeuralProcesses.models.utils.transformation import IdentityTransform

    def calc_regret(dataset: trieste_Dataset, global_maximum, time_scaling=1.0):
        """
        Calculate the regret
        """
        if opt_type == "so":
            return global_maximum - np.max(
                config.experimental_design.obj_func_form(dataset.observations).numpy()
            )
        else:
            existing_obs = np.concatenate(
                [
                    -config.experimental_design.obj_func_form(
                        dataset.observations
                    ).numpy(),
                    dataset.query_points[..., -1:].numpy() * time_scaling,
                ],
                axis=-1,
            )
            existing_obs_below_ref = existing_obs[
                np.all(existing_obs < config.experimental_design.ref_points, axis=-1)
            ]
            if np.size(existing_obs_below_ref) == 0:
                return global_maximum
            else:
                return global_maximum - Pareto(
                    existing_obs_below_ref
                ).hypervolume_indicator(config.experimental_design.ref_points)

    # extract some parameters from config
    opt_rng = random.PRNGKey(opt_seed)
    num_states = config.experimental_design.states_num
    total_num_of_trajs = config.experimental_design.num_traj_iter
    optimize_repeat = config.experimental_design.batch_size_change_times
    maximum_obs_per_traj = config.experimental_design.maximum_obs_per_traj
    init_cond_decision_dim = len(config.experimental_design.x0_lower_bound)

    current_regret = []
    initial_idx = []  # stor all the count where a new trajectory starts
    obs_count = 0

    # design of experiments
    # _, doe_rng = jax.random.split(
    #     opt_rng
    # )  # in gp based opt, the opt_rng is only used to generate the initial ode
    opt_rng, test_problem_rng, doe_rng, _ = random.split(opt_rng, 4)

    observer = partial(
        config.experimental_design.observer, 
        problem_rng=test_problem_rng, 
        time_scaling = config.data.args.time_scaling_coefficient
    )

    # extract global maximum value
    if (
        not config.experimental_design.fixed_problem
    ):  # the problem can be a generator, we hence also need to calculate corresponding regret
        if opt_type == "so":
            raise NotImplementedError
        elif opt_type == "mo":
            x0_search_space = Box(
                config.experimental_design.x0_lower_bound,
                config.experimental_design.x0_upper_bound,
            )
            t_search_space = Box(
                np.atleast_1d(config.experimental_design.t0),
                np.atleast_1d(config.experimental_design.t1),
            )
            obj = lambda xs, ts: config.experimental_design.obj_func_form(
                observer(
                    xs, ts, config.experimental_design.t0, config.experimental_design.t1
                )
            )
            pf, pf_inputs = sample_pareto_front_from_observer(
                x0_search_space * t_search_space, obj, 50, 
                time_scaling = config.data.args.time_scaling_coefficient, 
                initial_cond_mapper=config.experimental_design.initial_cond_mapper() if hasattr(config.experimental_design, 'initial_cond_mapper') else IdentityTransform()
            )
            ref_pts = get_reference_point(pf)
            reference_global_maximum = Pareto(pf).hypervolume_indicator(ref_pts)
        else:
            raise ValueError(f"opt_type: {opt_type} is not supported")
    else:
        reference_global_maximum = config.experimental_design.global_maximum_val

    doe_init_conds = uniform(
        key=doe_rng,  # note that the initial condition dim can be different from the states dim (e.g., in SIR problem)
        shape=(config.experimental_design.initial_traj_num, init_cond_decision_dim),
        minval=config.experimental_design.x0_lower_bound,
        maxval=config.experimental_design.x0_upper_bound,
    )  # [initial_traj_num, state_dim]

    opt_auxilary_infos = {
        "doe_init_conds": doe_init_conds,
        "opt_seed": opt_seed,
        "opt_type": opt_type,
        "reference_global_maximum": reference_global_maximum,
        "acq_init_smp": config.experimental_design.acq_mc_size,
        "acq_opt_par_num": config.experimental_design.acq_opt.acq_opt_parallel_num,
        "acq_opt_max_iter": config.experimental_design.acq_opt.acq_opt_max_iter,
    }

    # create initial dataset
    traj_dicts = {
        f"Traj{traj_idx}": SingleTrajectoryDataset(
            state_dim=init_loc.shape[-1],
            times=config.experimental_design.initial_obs_time,
            observations=observer(
                init_loc,
                config.experimental_design.initial_obs_time,
                config.experimental_design.t0,
                config.experimental_design.t1,
            ),
            initial_cond=init_loc,
        )
        for traj_idx, init_loc in enumerate(doe_init_conds)
    }

    np_datasets = NP_Dataset(traj_dicts)
    gp_datasets: trieste_Dataset = np_datasets.formalize_training_data_for_trieste(
        dtype=tf.float64
    )

    if opt_type == "so":
        traj_acquisition_builder = GreyBoxBatchMonteCarloExpectedImprovement(
            sample_size=config.experimental_design.acq_mc_size,
            obj_func_form=config.experimental_design.obj_func_form,
        )
    else:
        traj_acquisition_builder = GreyBoxBatchMonteCarloExpectedHypervolumeImprovement(
            sample_size=config.experimental_design.acq_mc_size,
            obj_func_form=config.experimental_design.obj_func_form,
            # reference_point_spec = config.experimental_design.ref_points,
        )

    traj_acquisition_rule = MinimumDelayConstrainedBatchEfficientGlobalOptimization(
        minimum_delta=config.experimental_design.time_delay,
        builder=traj_acquisition_builder,
        num_query_points=np.arange(
            maximum_obs_per_traj + 1 - optimize_repeat, maximum_obs_per_traj + 1, 1
        ),
        acq_optimizer_initial_smp_num=config.experimental_design.acq_opt.initial_smp_num,
        acq_optimizer_parallel_num=config.experimental_design.acq_opt.acq_opt_parallel_num,
        acq_optimizer_max_iter=config.experimental_design.acq_opt.acq_opt_max_iter,
    )
    current_regret.append(calc_regret(gp_datasets, reference_global_maximum))

    existing_initial_trajs = doe_init_conds.shape[0]
    for across_traj_opt_step in np.arange(
        existing_initial_trajs, total_num_of_trajs + existing_initial_trajs, 1
    ):
        pbar = tqdm(
            np.arange(1, maximum_obs_per_traj + 1, 1),
            total=maximum_obs_per_traj,
            initial=0,
            desc=f"trajectory: {across_traj_opt_step}",
        )
        logging.info(f"start optimize {across_traj_opt_step}th initial condition")

        if isinstance(gp_datasets, trieste_Dataset):
            gp_datasets = {OBJECTIVE: gp_datasets}
        if not isinstance(models, Mapping):
            models = {OBJECTIVE: models}

        # Identify the promising initial condition
        # try:
        if across_traj_opt_step == 1:
            for tag, model in models.items():
                gp_dataset = gp_datasets[tag]
                assert gp_dataset is not None
                model.update(gp_dataset)
                model.optimize(gp_dataset)
        with Timer() as initial_loc_opt_timer:
            points_or_stateful = traj_acquisition_rule.acquire(
                initial_loc_bounds=[
                    config.experimental_design.x0_lower_bound,
                    config.experimental_design.x0_upper_bound,
                ],
                t_bounds=[
                    config.experimental_design.t0,
                    config.experimental_design.t1,
                ],
                model=models,
                datasets=gp_datasets,
            )
        if callable(points_or_stateful):
            acquisition_state, query_points = points_or_stateful(acquisition_state)
        else:
            query_points = points_or_stateful
        identified_optimal_init_loc = tf.reshape(
            query_points[:init_cond_decision_dim], -1
        )
        last_obs_time = config.experimental_design.t0
        observer_output = observer(
            np.asarray(identified_optimal_init_loc.numpy()),
            np.atleast_1d(config.experimental_design.t0),
            config.experimental_design.t0,
            config.experimental_design.t1,
        )
        # add new innitial location in a new starting point
        np_datasets.append_new_traj(
            f"Traj{across_traj_opt_step}",
            SingleTrajectoryDataset(
                state_dim=doe_init_conds.shape[-1],
                times=np.asarray([config.experimental_design.t0]),
                observations=observer_output,
                initial_cond=np.asarray(identified_optimal_init_loc.numpy()),
            ),
        )
        gp_datasets: trieste_Dataset = np_datasets.formalize_training_data_for_trieste(
            dtype=tf.float64
        )

        obs_count += 1
        current_regret.append(calc_regret(gp_datasets, reference_global_maximum))
        log_iteration(
            obs_count,
            identified_optimal_init_loc.numpy(),
            last_obs_time,
            observer_output,
            initial_loc_opt_timer.time,
            current_regret[-1],
            log_file_path=os.path.join(
                workdir,
                "opt",
                f"{opt_type}",
                f"optimization_log_{str(opt_seed)}.csv",
            ),
            auxilary_info=opt_auxilary_infos,
        )

        initial_idx.append(obs_count)
        maximum_remainin_obs = maximum_obs_per_traj
        # within trajectory optimization
        for (
            within_traj_opt_step
        ) in pbar:  # np.arange(1, total_num_of_obs_per_traj + 1, 1):
            logging.info(
                f"start optimize {within_traj_opt_step}th obs schedule within {across_traj_opt_step}th trajectory"
            )
            if isinstance(gp_datasets, trieste_Dataset):
                gp_datasets = {OBJECTIVE: gp_datasets}

            for tag, model in models.items():
                gp_dataset = gp_datasets[tag]
                assert gp_dataset is not None
                model.update(gp_dataset)
                model.optimize(gp_dataset)

            if np.size(maximum_remainin_obs) == 0 or maximum_remainin_obs < 1:
                break  # stop trajectory optimization

            if within_traj_opt_step == 1:
                query_points = query_points[
                    init_cond_decision_dim:
                ]  # we by pass this optimization step
                within_traj_opt_time = np.inf
            else:
                with Timer() as within_traj_opt_timer:
                    points_or_stateful = traj_acquisition_rule.acquire(
                        initial_loc=identified_optimal_init_loc,
                        t_bounds=[last_obs_time, config.experimental_design.t1],
                        model=models,
                        last_obs_time=last_obs_time,
                        datasets=gp_datasets,
                    )
                within_traj_opt_time = within_traj_opt_timer.time
                if callable(points_or_stateful):
                    acquisition_state, query_points = points_or_stateful(
                        acquisition_state
                    )
                else:
                    query_points = points_or_stateful

            # conduct the time consuming observation
            last_obs_time = np.squeeze(np.asarray(query_points[0].numpy()))
            observer_output = observer(
                np.asarray(identified_optimal_init_loc.numpy()),
                np.atleast_1d(last_obs_time),
                config.experimental_design.t0,
                config.experimental_design.t1,
            )
            # update dataset
            np_datasets.append_obs_within_traj(
                f"Traj{across_traj_opt_step}",
                np.atleast_1d(last_obs_time),
                observer_output,
            )
            gp_datasets: trieste_Dataset = (
                np_datasets.formalize_training_data_for_trieste(dtype=tf.float64)
            )
            obs_count += 1
            current_regret.append(calc_regret(gp_datasets, reference_global_maximum))
            log_iteration(
                obs_count,
                identified_optimal_init_loc.numpy(),
                last_obs_time,
                observer_output,
                within_traj_opt_time,
                current_regret[-1],
                log_file_path=os.path.join(
                    workdir,
                    "opt",
                    f"{opt_type}",
                    f"optimization_log_{str(opt_seed)}.csv",
                ),
            )

            # update num of remaining query points within the trajectory
            maximum_remainin_obs = np.floor(
                (config.experimental_design.t1 - last_obs_time)
                / config.experimental_design.time_delay
            ).astype(np.int32)
            # next_num_query_points = np.arange(
            #     maximum_remainin_obs + 1 - optimize_repeat,
            #     maximum_remainin_obs + 1,
            #     1,
            # )
            # next_num_query_points = next_num_query_points[
            #     next_num_query_points >= 1
            # ]  # filter out zero/negative num query points
            # the issue is that maximum_remainin_obs keep decreasing but optimize_repeat remains constant
            next_num_query_points = list(
                range(
                    int(np.ceil((maximum_remainin_obs) / 2)),
                    int(maximum_remainin_obs + 1),
                )
            )
            traj_acquisition_rule.update_num_query_points(next_num_query_points)
            pbar.total = int(within_traj_opt_step + maximum_remainin_obs)
        # reset num of remaining query points within the trajectory
        traj_acquisition_rule.update_num_query_points(
            np.arange(
                config.experimental_design.maximum_obs_per_traj + 1 - optimize_repeat,
                config.experimental_design.maximum_obs_per_traj + 1,
                1,
            )
        )
        with open(
            os.path.join(
                workdir,
                "opt",
                f"{opt_type}",
                f"optimization_seed_{str(opt_seed)}_sampled_traj.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(np_datasets.traj_dicts, f)
