"""
Model comparison of SANODEP vs NP on meta learn of ODE distributions
"""

'''
This script aims to plot the model comparison, in a way such that the legend can be unified
'''
import os
import jax
import tensorflow as tf
from jax import numpy as np
from matplotlib import pyplot as plt
from ml_collections import ConfigDict
from NeuralProcesses.train.optimizer import create_optimizer
from NeuralProcesses.models import build
from NeuralProcesses.data import datasets
from ml_collections import ConfigDict
from flax.training import train_state
from flax.training import checkpoints
import tensorflow as tf
import logging
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import tensorflow as tf


def plot_interpolating_and_forecasting_for_models():
    """
    Plot GP ODE vector field
    1st line: GP samples 1
    2nd line: GP samples 2
    3rd line: vector field and Sample Trajectories
    4th line: model prediction
    """
  
    # plt.figure()
    # _, axs = plt.subplots(nrows=2, ncols=5, figsize=(9, 3.5))
    import importlib.util

    def load_config_from_py(config_file):
        spec = importlib.util.spec_from_file_location("config", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.get_config()  # Call the get_config function

    # for pb_idx, pb in enumerate(pb_list):
        # print(f'work on {pb}')
    np_config_file = os.path.join('exps/cfgs/', 'gp_ode_2d', f'np.py')
    np_config = load_config_from_py(np_config_file)
    np_config.data.foracsting_problem_prob = 0.5 # this does not matter that much, as we will do forecasting and interpolating anyhow
    np_config.model.t1 = 1.0
    np_config.data.args.t_range = (0, 1.0)
    np_config.data.args.time_scaling_coefficient = 10.0
    # get the abs path of current script
    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    np_workdir = os.path.join(dir_path, 'experiments/gp_ode_2d/np/forcast_prob0.5/seed_0' )
    fig, axs = plot_meta_learn_model(np_config, np_workdir, title='NP prediction', plot_axis_row = 4)    

    sanodep_config_file = os.path.join('exps/cfgs/', 'gp_ode_2d', f'sanodep.py')
    sanodep_config = load_config_from_py(sanodep_config_file)
    sanodep_config.data.foracsting_problem_prob = 0.5 # this does not matter that much, as we will do forecasting and interpolating anyhow
    sanodep_config.model.t1 = 1.0
    sanodep_config.data.args.t_range = (0, 1.0)
    sanodep_config.data.args.time_scaling_coefficient = 10.0
    # sanodep_workdir = f'exps/experiments/gp_ode_2d/sanodep/forcast_prob0.0/seed_0'
    sanodep_workdir = os.path.join(dir_path, 'experiments/gp_ode_2d/sanodep/forcast_prob0.5/seed_0' )
    fig, axs = plot_meta_learn_model(sanodep_config, sanodep_workdir, title='SANODEP prediction', plot_axis_row = 5, existing_axs = axs, existing_fig=fig)
    
    # plot_tensorboard_log(fig, axs)
    # config_sanodep.data.foracsting_problem_prob = 0.5 # this does not matter that much, as we will do forecasting and interpolating anyhow
    # config_file_sanodep = os.path.join('exps/cfgs/', problem_name_mapping[pb], f'sanodep.py')
    # config_sanodep = load_config_from_py(config_file_sanodep)
    # legend = plt.legend(frameon=False, fontsize=14)
    # So far, nothing special except the managed prop_cycle. Now the trick:
    lines_labels = [ax.get_legend_handles_labels() for ax in axs.flat]
    lines, labels = zip(*lines_labels)
    # Combine lines and labels into pairs
    pairs = list(zip(lines, labels))
    # Initialize an empty dictionary

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.3, top=0.95, wspace=0.2, hspace=0.36)
    # save it to the current script directory
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    fig.savefig(os.path.join(script_dir, 'experiments', 'figs', 'gp_ode_sample_and_model_comparisons.png'), dpi=300, bbox_inches='tight')


def plot_meta_learn_model(config: ConfigDict, workdir: str, plot_axis_row, title, existing_axs: bool = None, existing_fig: bool = None):
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    assert isinstance(config.seed, int)  # type check
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    model, initial_params, rng = build.init_model(init_rng, config)
    # instantiate optimizers

    # create training state
    if "create_optimizer" in config.training:
        optimizer = config.training["create_optimizer"](config)
    else:
        optimizer = create_optimizer(config)

    training_state = train_state.TrainState.create(
        apply_fn=model.apply, params=initial_params, tx=optimizer
    )
    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Resume training when intermediate checkpoints are detected
    training_state = checkpoints.restore_checkpoint(checkpoint_dir, training_state)
    # `state.step` is JAX integer on the GPU/TPU devices
    initial_step = int(training_state.step)
    trained_epoch = initial_step // config.training.optimizer.args.num_steps_per_epoch
    need_train_epoch = config.training.num_epochs - trained_epoch

    # Build data iterators
    rng, shuffle_rng = jax.random.split(rng)
    data, dataset_inst = datasets.get_dataset(config)
    plot_datasets = dataset_inst.plot_datasets
    if existing_axs is None:
        fig, axs = dataset_inst.plot_handles
    else:
        axs = existing_axs
        fig = existing_fig

    logging.info("Starting training loop at step %d." % (initial_step,))


    this_sample_dir = (
        sample_dir  # os.path.join(, "epoch_{}".format(current_epoch))
    )
    tf.io.gfile.makedirs(this_sample_dir)
    axs = meta_learn_model_interploating_plot(
        config=config,
        model=model,
        rng=rng,
        training_state=training_state,
        dataset_inst=dataset_inst,
        current_epoch = trained_epoch, 
        aux_batch=(*plot_datasets, np.zeros_like(plot_datasets[1])),
        this_sample_dir=this_sample_dir, 
        plot_axs = axs, 
        sample_system_number = 6, 
        plot_axis_row = plot_axis_row, 
        title=title)
    return fig, axs
    # fig.savefig('gp_ode_sample_and_model_comparisons.png', dpi=300, bbox_inches='tight')


def meta_learn_model_interploating_plot(**kwargs):
    """
    Here we make two plot, one is for interpolating, the other one is for forcasting

    We will make use of the auxilary data to make the plot
    """
    from jax import vmap
    # from matplotlib import pyplot as plt

    # plt.rcParams['text.usetex'] = True
    from NeuralProcesses.data.datasets import get_data_preprocessor

    config = kwargs["config"]
    # we use the negative log likelihood loss based auxilary data to make the plot
    data_t, data_x, data_params = kwargs["aux_batch"] # kwargs["aux_batch"]["NLL"]
    rng = kwargs["rng"]
    model = kwargs["model"]
    this_sample_dir = kwargs["this_sample_dir"]
    current_epoch = kwargs["current_epoch"]
    sample_system_number = kwargs['sample_system_number']
    traj_size = data_x.shape[1]
    training_state = kwargs["training_state"]
    dataset_inst = kwargs["dataset_inst"]
    plot_axis_row = kwargs["plot_axis_row"]
    t0 = config.model.t0
    t1 = config.model.t1
    # why not directly make use of data_preprocessor?
    pre_processor = get_data_preprocessor(dataset_inst, config)
    # pre_processor(dataset_inst, config)
    aux_data_batch = (data_t, data_x, data_params)
    processed_data, rng = pre_processor(
        aux_data_batch, rng, known_traj_range=config.data.known_traj_range
    )
    # variables = {"params": training_state.params}
    (
        data_t,
        data_x,
        data_params, 
        context_mask,
        target_mask,
        ctx_mask_with_new_traj_obs,
        ctx_mask_with_new_traj_target_mask,
        target_initial_cond_mask,
        target_mask_unknown_traj,
        known_trajectory,
    ) = processed_data
    # we do not use any sorting here since it does not make too much sense
    data_t = np.squeeze(data_t, axis=-1)

    # calculate the model prediction for the given data
    batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt_x0, mask_ctx_x0, mask_ctx_with_new_traj: model.apply(
        training_state.params,
        t_context=tctx,
        x_context=x_ctx,
        t_target=t_tgt,
        sample_rng=rng,
        sample_size=config.snap_shot_sampling_cfg.model_sample_size,
        x_target=x_tgt,
        training=False,
        target_initial_cond_mask=mask_tgt_x0,
        ctx_mask_with_new_traj_obs=mask_ctx_x0,
        ctx_mask_with_new_traj_target_mask=mask_ctx_with_new_traj,
        solver="Dopri5",
        t0=t0,
        t1=t1,
    )

    # make the plot:
    # for each of the dynamic system, we make 2 plots, one for interpolating, the other one for forcasting
    # we want to compare the model behavior on the same trajectory, starting at the same initial condition and hence, 
    # We use the existing ctx_mask_with_new_traj_obs for interpolating, but we create a new ctx_mask_with_new_traj_obs
    # for forcasting starting at the same initial condition 
    # print(f'sample_system_number: {sample_system_number}')
    for system_idx in range(sample_system_number):
        # print(f'work on system {system_idx}')
        # find the very beginning interpolating trajectory and forcasting trajectory
        _interpolating_traj_idx = None
        # print(ctx_mask_with_new_traj_obs[0, -1, -1])
        for traj_idx in range(traj_size):
            if np.sum(ctx_mask_with_new_traj_obs[system_idx, traj_idx, traj_idx]) != 1:
                _interpolating_traj_idx = traj_idx
                break
        if _interpolating_traj_idx is None:
            raise ValueError("No interpolating trajectory found")
        # model prediction for interpolating
        x_pred_f, x_pred_sigma = vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_initial_cond_mask,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
        )  # [batch_size, traj_size, num_samples, num_points, output_dim], [batch_size, traj_size, num_samples, num_points, output_dim]

        # import matplotlib.pyplot as plt

        # plt.rcParams['text.usetex'] = True
        import matplotlib.cm as cm

        # Use a color map
        # colors = cm.get_cmap("tab10")
        # colors = cm.viridis(np.linspace(0, 1, data_x.shape[-1])) 
        colors = cm.viridis(np.linspace(0, 0.9, 3)) # cm.viridis(np.linspace(0, 0.9, data_x.shape[-1]))  # Use a subset of the 'viridis' color map
        colors = colors[::-1]  # Reverse the color map
        axs = kwargs['plot_axs']
        model_pred_label = False
        # Use a serif font
        plt.rcParams["font.family"] = "serif"
        axs[plot_axis_row, system_idx].tick_params(axis='both', which='major', labelsize=6)  # Increase tick size
        for mu, sigma in zip(
            x_pred_f[system_idx, _interpolating_traj_idx],
            x_pred_sigma[system_idx, _interpolating_traj_idx],
        ):  
            # print(f'_interpolating_traj_idx: {_interpolating_traj_idx}')
            # print(x_pred_f.shape)
            # print(mu.shape)
            # print(sigma.shape)
            for state_idx in range(mu.shape[-1]):
                axs[plot_axis_row, system_idx].fill_between(
                    data_t[system_idx, _interpolating_traj_idx],
                    np.squeeze(mu[..., state_idx] - 1.96 * sigma[..., state_idx]),
                    np.squeeze(mu[..., state_idx] + 1.96 * sigma[..., state_idx]),
                    color=colors[state_idx],
                    alpha=0.01,
                )

            # markers = ["o", "v", "^", "<", ">", "s", "p", "*"]
            for state_idx in range(mu.shape[-1]):
                axs[plot_axis_row, system_idx].plot(
                    data_t[system_idx, _interpolating_traj_idx],
                    mu[..., state_idx],
                    color=colors[state_idx],
                    linewidth=0.5,
                    label=f"Predicted State {state_idx + 1} Value" if not model_pred_label else "",
                )
            model_pred_label = True

        new_context_label_added = False
        context_label_added = False
        for traj_idx in range(traj_size):
            if traj_idx == _interpolating_traj_idx:
                for state_idx in range(mu.shape[-1]):
                    axs[plot_axis_row, system_idx].scatter(
                        data_t[system_idx, traj_idx][
                            ctx_mask_with_new_traj_obs[
                                system_idx, _interpolating_traj_idx, traj_idx
                            ]
                        ],
                        data_x[system_idx, traj_idx][
                            ctx_mask_with_new_traj_obs[
                                system_idx, _interpolating_traj_idx, traj_idx
                            ]
                        ][..., state_idx],
                        marker="^",
                        color='k',
                        s=15,
                        zorder=1000,
                        edgecolors="w",  # Add white edge
                        linewidths=0.4,  # Adjust the width of the edge
                        label=(
                            r'$\mathcal{T}_{new}^{\mathbb{C}}$' if not new_context_label_added else ""
                        ),  # Add label for context data
                    )
                    new_context_label_added = True
                    axs[plot_axis_row, system_idx].plot(
                        data_t[system_idx, traj_idx],
                        data_x[system_idx, traj_idx][..., state_idx],
                        "--",
                        color=colors[state_idx],
                        linewidth=1.5,
                        zorder=40,
                        label=f"Real State {state_idx + 1} Value",  # Add label for state value
                    )
            else:
                context_other_traj_label_added = False
                for state_idx in range(mu.shape[-1]):
                    axs[plot_axis_row, system_idx].scatter(
                        data_t[system_idx, traj_idx][
                            ctx_mask_with_new_traj_obs[
                                system_idx, _interpolating_traj_idx, traj_idx
                            ]
                        ],
                        data_x[system_idx, traj_idx][
                            ctx_mask_with_new_traj_obs[
                                system_idx, _interpolating_traj_idx, traj_idx
                            ]
                        ][..., state_idx],
                        s=4,
                        color=colors[state_idx],
                        zorder=50,
                        alpha=0.4,
                        edgecolors='k',  # Add white edge
                        linewidths=0.4,  # Adjust the width of the edge
                        label=(
                            r'$\mathcal{T}^{\mathbb{C}}$' if not context_label_added else ""
                        ),  # Add label for context data
                    )
                context_label_added = True
        axs[plot_axis_row, system_idx].set_title(kwargs['title'], fontsize=8)
        if plot_axis_row == 5:
            axs[plot_axis_row, system_idx].set_xlabel("Time", fontsize=8)
        if plot_axis_row >= 2:
            axs[plot_axis_row, system_idx].grid(True)
        # Remove the box around the legend
        # legend = plt.legend(frameon=False, fontsize=14)
        axs[plot_axis_row, system_idx].set_xlim(t0, t1)
        axs[plot_axis_row, system_idx].set_ylim(*axs[3, system_idx].get_ylim())
        # plt.tight_layout()
        # plt.savefig(
        #     os.path.join(this_sample_dir, f"{config.model.name}_{config.data.dataset_name}_forcast_prob{config.data.foracsting_problem_prob}_Interpolating_epoch{current_epoch}_on_system{system_idx}.png"),
        #     dpi=300,  # Increase dpi for higher resolution
        # )
    return axs



if __name__ == '__main__':
    plot_interpolating_and_forecasting_for_models()