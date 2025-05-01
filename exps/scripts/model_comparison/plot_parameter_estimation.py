'''
This script aims to plot the model comparison, in a way such that the legend can be unified
'''
import os
import jax
import tensorflow as tf
from jax import numpy as np
from matplotlib import pyplot as plt
from ml_collections import ConfigDict
from NeuralProcesses.training.optimizer import create_optimizer
from NeuralProcesses.models import build
from NeuralProcesses.data import datasets
from ml_collections import ConfigDict
from flax.training import train_state
from flax.training import checkpoints
import tensorflow as tf
import logging
import os


def plot_parameter_estimation_from_models(seed, tr_fc_pb):
    # pb_list = ['LV', 'Brusselator', 'FitzHug', 'Selkov', 'SIR', 'LV3D']
    problem_name_mapping = {'LV': 'lotka_voterra', 'Brusselator': 'brusselator', 'FitzHug': 'FitzHugh_Nagumo', 'Selkov': 'selkov', 'SIR': 'sir_problem', 'LV3D': 'three_dim_lotka_voterra'}
    pb = 'LV'
    
    
    plt.figure()
    _, axs = plt.subplots(nrows=1, ncols=6, figsize=(9, 2))
    import importlib.util

    def load_config_from_py(config_file):
        spec = importlib.util.spec_from_file_location("config", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.get_config()  # Call the get_config function

    # plot the original SANODEP
    config_file = os.path.join('exps/cfgs/', problem_name_mapping[pb], f'sanodep.py')
    config = load_config_from_py(config_file)
    config.data.foracsting_problem_prob = 0.5 # this does not matter that much, as we will do forecasting and interpolating anyhow
    workdir = f'exps/experiments/{problem_name_mapping[pb]}/sanodep/forcast_prob{tr_fc_pb}/seed_{seed}' if pb != 'LV' else f'exps/experiments/{problem_name_mapping[pb]}/model_comparison/sanodep/forcast_prob0.5/seed_{seed}'
    debug_plot_meta_learn_model(config, workdir, axs[0], r'SANODEP-$\lambda = 0.5$')

    # Plot the physics enhanced SANODEP
    config_file = os.path.join('exps/cfgs/', problem_name_mapping[pb], f'grey_box_LV_sanodep.py')
    config = load_config_from_py(config_file)
    config.data.foracsting_problem_prob = 0.5 # this does not matter that much, as we will do forecasting and interpolating anyhow

    workdir = f'exps/experiments/{problem_name_mapping[pb]}/grey_box_LV_sanodep/forcast_prob{tr_fc_pb}/seed_{seed}' if pb != 'LV' else f'exps/experiments/{problem_name_mapping[pb]}/model_comparison/grey_box_LV_sanodep/forcast_prob0.5/seed_{seed}'
    debug_plot_meta_learn_model(config, workdir, axs, r'PI-SANODEP-$\lambda = 0.5$')

    # Plot the parameter estimation result


    # legend = plt.legend(frameon=False, fontsize=14)
    # So far, nothing special except the managed prop_cycle. Now the trick:
    lines_labels = [ax.get_legend_handles_labels() for ax in axs.flat]
    lines, labels = zip(*lines_labels)
    # Combine lines and labels into pairs
    pairs = list(zip(lines, labels))
    # Initialize an empty dictionary
    unique_dict = {}

    for lines, labels in pairs:
        for _line, _label in zip(lines, labels):
            # If the label is not in the dictionary, add the line and label to the dictionary
            if _label!=[]:
                # print(_label)
                if _label not in unique_dict:
                    unique_dict[_label] = _line

    # Convert the dictionary back to a list of tuples
    unique_pairs = list(unique_dict.items())

    # Unpack unique lines and labels
    unique_labels, unique_lines  = zip(*unique_pairs)

    # Create legend with unique lines and labels
    # axs.legend(unique_labels, unique_lines, loc='lower center')
    # Get the current figure
    fig = plt.gcf()

    # Create legend with unique lines and labels for the entire figure
    # Create legend with unique lines and labels for the entire figure
    # fig.legend(unique_lines, unique_labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=9, fontsize=6, markerscale=2)

    # Adjust the subplots to create space for the legend
    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom parameter to create more space

    # fig = plt.gcf()  # Get the current figure
    fig.legend(unique_lines, unique_labels, loc='lower center', ncol=7, fontsize=8, frameon=False)
    # Finally, the legend (that maybe you'll customize differently)
    # plt.legend(lines_dict.values(), lines_dict.keys(), loc='lower center', ncol=4)
    # axs.set_xlim(config.model.t0, config.model.t1)
    # plt.tight_layout(pad=0.9, w_pad=0.5, h_pad=0.1)
    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
    # plt.tight_layout()
    axs[0].set_ylabel('Value')
    axs[2].set_ylabel('Density')
    axs[0].set_ylim(0, 5)
    axs[1].set_ylim(0, 5)
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.27, top=0.85, wspace=0.2, hspace=0.2)
    plt.savefig(
        'param_est.png',
        dpi=500,  # Increase dpi for higher resolution
    )


def debug_plot_meta_learn_model(config: ConfigDict, workdir: str, axs, title):
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    assert isinstance(config.seed, int)  # type check
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    model, init_model_state, initial_params, rng = build.init_model(init_rng, config)
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
    # the order of shuffle, batch and repeat shall follows strictly as states here: https://stackoverflow.com/questions/49915925/output-differences-when-changing-order-of-batch-shuffle-and-repeat#:~:text=Best%20Ordering%3A&text=For%20batches%20to%20be%20different,are%20unique%2C%20unlike%20the%20other.
    if (
        config.data.args.generator is True
    ):  # if dataset is gen from generator, no need to shuffle and batch (assumed it has been done in dataset level)
        pass
    else:
        data = (
            data.shuffle(
                buffer_size=config.data.shuffle_buffer_size,
                seed=int(shuffle_rng[0]),
                reshuffle_each_iteration=True,
            )
            .batch(batch_size=config.training.batch_size)
            .repeat(need_train_epoch)
        )


    # auxilary data to supervise the model performance during training
    if config.data.args.get("aux"):
        aux_datsets = datasets.get_aux_datasets(dataset_inst, config)

    logging.info("Starting training loop at step %d." % (initial_step,))


    this_sample_dir = (
        sample_dir  # os.path.join(, "epoch_{}".format(current_epoch))
    )
    tf.io.gfile.makedirs(this_sample_dir)
    meta_learn_model_interploating_plot(
        config=config,
        model=model,
        rng=rng,
        training_state=training_state,
        dataset_inst=dataset_inst,
        current_epoch = trained_epoch, 
        aux_batch=aux_datsets,
        this_sample_dir=this_sample_dir, 
        plot_axs = axs, 
        title=title)



def meta_learn_model_interploating_plot(**kwargs):
    """
    Here we make two plot, one is for interpolating, the other one is for forcasting

    We will make use of the auxilary data to make the plot
    """
    from jax import vmap
    from matplotlib import pyplot as plt

    # plt.rcParams['text.usetex'] = True
    from NeuralProcesses.data.datasets import get_data_preprocessor

    config = kwargs["config"]
    # we use the negative log likelihood loss based auxilary data to make the plot
    data_t, data_x, sampled_params = kwargs["aux_batch"]["NLL"]
    alpha, beta, delta, gamma = sampled_params
    rng = kwargs["rng"]
    model = kwargs["model"]
    this_sample_dir = kwargs["this_sample_dir"]
    current_epoch = kwargs["current_epoch"]
    sample_system_number = config.snap_shot_sampling_cfg.dynamics_sample_number
    traj_size = data_x.shape[1]
    training_state = kwargs["training_state"]
    dataset_inst = kwargs["dataset_inst"]
    t0 = config.model.t0
    t1 = config.model.t1
    # why not directly make use of data_preprocessor?
    pre_processor = get_data_preprocessor(dataset_inst, config)
    # pre_processor(dataset_inst, config)
    aux_data_batch = (data_t, data_x, data_params)
    processed_data, rng = pre_processor(
        aux_data_batch, rng, known_traj_range=config.data.known_traj_range
    )
    variables = {"params": training_state.params}
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
        variables,
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
    for system_idx in range(sample_system_number):
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
        if kwargs['config'].model.name == 'SANODEP':
            x_pred_f, x_pred_sigma = vmap(batch_model_apply)(
                data_t,
                data_x,
                data_t,
                data_x,
                target_initial_cond_mask,
                ctx_mask_with_new_traj_obs,
                ctx_mask_with_new_traj_target_mask,
            )  # [batch_size, traj_size, num_samples, num_points, output_dim], [batch_size, traj_size, num_samples, num_points, output_dim] 
        else:        
            x_pred_f, x_pred_sigma, params_est = vmap(batch_model_apply)(
                data_t,
                data_x,
                data_t,
                data_x,
                target_initial_cond_mask,
                ctx_mask_with_new_traj_obs,
                ctx_mask_with_new_traj_target_mask,
            )  # [batch_size, traj_size, num_samples, num_points, output_dim], [batch_size, traj_size, num_samples, num_points, output_dim]

        import matplotlib.pyplot as plt

        # plt.rcParams['text.usetex'] = True
        import matplotlib.cm as cm

        # Use a color map
        # colors = cm.get_cmap("tab10")
        # colors = cm.viridis(np.linspace(0, 1, data_x.shape[-1])) 
        colors = cm.viridis(np.linspace(0, 0.9, 3)) # cm.viridis(np.linspace(0, 0.9, data_x.shape[-1]))  # Use a subset of the 'viridis' color map
        colors = colors[::-1]  # Reverse the color map
        all_axes = kwargs['plot_axs']
        model_pred_label = False
        # Use a serif font
        plt.rcParams["font.family"] = "serif"
        # all_axes.tick_params(axis='both', which='major', labelsize=6)  # Increase tick size
        if kwargs['config'].model.name == 'SANODEP':
            axs = all_axes
        else:
            axs = all_axes[1]
        for mu, sigma in zip(
            x_pred_f[system_idx, _interpolating_traj_idx],
            x_pred_sigma[system_idx, _interpolating_traj_idx],
        ):  
            # print(f'_interpolating_traj_idx: {_interpolating_traj_idx}')
            # print(x_pred_f.shape)
            # print(mu.shape)
            # print(sigma.shape)
            for state_idx in range(mu.shape[-1]):
                axs.fill_between(
                    data_t[system_idx, _interpolating_traj_idx],
                    np.squeeze(mu[..., state_idx] - 1.96 * sigma[..., state_idx]),
                    np.squeeze(mu[..., state_idx] + 1.96 * sigma[..., state_idx]),
                    color=colors[state_idx],
                    alpha=0.01,
                )

            # markers = ["o", "v", "^", "<", ">", "s", "p", "*"]
            for state_idx in range(mu.shape[-1]):
                axs.plot(
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
                    axs.scatter(
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
                    axs.plot(
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
                    axs.scatter(
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

        # Remove the box around the legend
        # legend = plt.legend(frameon=False, fontsize=14)
        axs.set_xlim(t0, t1)
        axs.set_title(kwargs['title'], fontsize=8)

        if kwargs['config'].model.name != 'SANODEP':
            mu_alpha, sigma_alpha, mu_beta, sigma_beta, mu_gamma,  sigma_gamma,  mu_delta, sigma_delta = params_est
            mu_alpha = mu_alpha[system_idx, traj_idx]
            sigma_alpha = sigma_alpha[system_idx, traj_idx]
            mu_beta = mu_beta[system_idx, traj_idx]
            sigma_beta = sigma_beta[system_idx, traj_idx]
            mu_gamma = mu_gamma[system_idx, traj_idx]
            sigma_gamma = sigma_gamma[system_idx, traj_idx]
            mu_delta = mu_delta[system_idx, traj_idx]
            sigma_delta = sigma_delta[system_idx, traj_idx]
            from scipy.stats import lognorm
            scale_alpha = np.exp(mu_alpha)
            s_alpha = np.sqrt(sigma_alpha)

            # Generate x values
            x_alpha = np.linspace(0, 2, 100) # np.linspace(lognorm.ppf(0.001, s_alpha, scale=scale_alpha),
                      #       lognorm.ppf(0.999, s_alpha, scale=scale_alpha), 100)

            # Generate y values (lognormal PDF)
            y_alpha = lognorm.pdf(x_alpha, s_alpha, scale=scale_alpha)
            all_axes[2].plot(x_alpha, y_alpha)
            ylims = all_axes[2].get_ylim()
            all_axes[2].vlines(alpha[system_idx, traj_idx], ylims[0], ylims[1], color='r', linestyle='--', linewidth=0.8)
            all_axes[2].set_yticks([])
            all_axes[2].set_title(r'$\alpha$')
            all_axes[2].set_xlim(0, 2)

            scale_beta = np.exp(mu_beta)
            s_beta = np.sqrt(sigma_beta)
            x_beta = np.linspace(0, 2, 100) # np.linspace(lognorm.ppf(0.001, s_beta, scale=scale_beta),
                     #        lognorm.ppf(0.999, s_beta, scale=scale_beta), 100)

            # Generate y values (lognormal PDF)
            y_beta = lognorm.pdf(x_beta, s_beta, scale=scale_beta)
            all_axes[3].plot(x_beta, y_beta)
            ylims = all_axes[3].get_ylim()
            all_axes[3].vlines(beta[system_idx, traj_idx], ylims[0], ylims[1], color='r', linestyle='--', linewidth=0.8)
            all_axes[3].set_yticks([])
            all_axes[3].set_title(r'$\beta$')
            all_axes[3].set_xlim(0, 2)

            scale_delta = np.exp(mu_delta)
            s_delta = np.sqrt(sigma_delta)
            x_delta = np.linspace(0, 2, 100) # np.linspace(lognorm.ppf(0.001, s_delta, scale=scale_delta),
                      #       lognorm.ppf(0.999, s_delta, scale=scale_delta), 100)

            # Generate y values (lognormal PDF)
            y_delta = lognorm.pdf(x_delta, s_delta, scale=scale_delta)
            all_axes[4].plot(x_delta, y_delta)
            ylims = all_axes[4].get_ylim()
            all_axes[4].vlines(delta[system_idx, traj_idx], ylims[0], ylims[1], color='r', linestyle='--', linewidth=0.8)
            all_axes[4].set_yticks([])
            all_axes[4].set_title(r'$\delta$')
            all_axes[4].set_xlim(0, 2)

            scale_gamma = np.exp(mu_gamma)
            s_gamma = np.sqrt(sigma_gamma)
            # x_gamma = np.linspace(lognorm.ppf(0.001, s_gamma, scale=scale_gamma),
            #                 lognorm.ppf(0.999, s_gamma, scale=scale_gamma), 100)
            x_gamma = np.linspace(0, 2, 100)

            # Generate y values (lognormal PDF)
            y_gamma = lognorm.pdf(x_gamma, s_gamma, scale=scale_gamma)
            all_axes[5].plot(x_gamma, y_gamma)
            ylims = all_axes[5].get_ylim()
            all_axes[5].vlines(gamma[system_idx, traj_idx], ylims[0], ylims[1], color='r', linestyle='--', linewidth=0.8)
            all_axes[5].set_yticks([])
            all_axes[5].set_title(r'$\gamma$')
            all_axes[5].set_xlim(0, 2)

            # get the ground truth 
        # plt.tight_layout()
        # plt.savefig(
        #     os.path.join(this_sample_dir, f"{config.model.name}_{config.data.dataset_name}_forcast_prob{config.data.foracsting_problem_prob}_Interpolating_epoch{current_epoch}_on_system{system_idx}.png"),
        #     dpi=300,  # Increase dpi for higher resolution
        # )


def meta_learn_model_forecasting_plot(**kwargs):
    """
    Here we make two plot, one is for interpolating, the other one is for forcasting

    We will make use of the auxilary data to make the plot
    """
    from jax import vmap
    from matplotlib import pyplot as plt

    # plt.rcParams['text.usetex'] = True
    from NeuralProcesses.data.datasets import get_data_preprocessor

    config = kwargs["config"]
    # we use the negative log likelihood loss based auxilary data to make the plot
    data_t, data_x, data_params = kwargs["aux_batch"]["NLL"]
    rng = kwargs["rng"]
    model = kwargs["model"]
    this_sample_dir = kwargs["this_sample_dir"]
    current_epoch = kwargs["current_epoch"]
    sample_system_number = config.snap_shot_sampling_cfg.dynamics_sample_number
    traj_size = data_x.shape[1]
    training_state = kwargs["training_state"]
    dataset_inst = kwargs["dataset_inst"]
    t0 = config.model.t0
    t1 = config.model.t1
    # why not directly make use of data_preprocessor?
    pre_processor = get_data_preprocessor(dataset_inst, config)
    # pre_processor(dataset_inst, config)
    aux_data_batch = (data_t, data_x, data_params)
    processed_data, rng = pre_processor(
        aux_data_batch, rng, known_traj_range=config.data.known_traj_range
    )
    variables = {"params": training_state.params}
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
        variables,
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
    for system_idx in range(sample_system_number):
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

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # plt.rcParams['text.usetex'] = True
        # Use a color map
        # colors = cm.get_cmap("tab10")
        # colors = cm.viridis(np.linspace(0, 1, data_x.shape[-1])) 
        colors = cm.viridis(np.linspace(0, 0.9, 3)) # cm.viridis(np.linspace(0, 0.9, data_x.shape[-1]))  # Use a subset of the 'viridis' color map
        colors = colors[::-1]  # Reverse the color map

        # we now again do model prediction but for forcasting\
        ctx_mask_with_new_traj_initial_cond_only = np.logical_and(ctx_mask_with_new_traj_obs, \
                                                                  np.repeat(np.expand_dims(target_initial_cond_mask, axis=1), ctx_mask_with_new_traj_obs.shape[1], axis=1))
        _forcasting_traj_idx = _interpolating_traj_idx
        # model prediction for forcasting
        x_pred_f, x_pred_sigma = vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_initial_cond_mask,
            ctx_mask_with_new_traj_initial_cond_only,
            ctx_mask_with_new_traj_target_mask,
        )  # [batch_size, traj_size, num_samples, num_points, output_dim], [batch_size, traj_size, num_samples, num_points, output_dim]
        # make the forcasting plot

        axs = kwargs['plot_axs']
        model_pred_label = False
        
        axs.tick_params(axis='both', which='major', labelsize=4)  # Increase tick size
        # Use a serif font
        plt.rcParams["font.family"] = "serif"

        for mu, sigma in zip(
            x_pred_f[system_idx, _forcasting_traj_idx],
            x_pred_sigma[system_idx, _forcasting_traj_idx],
        ):
            for state_idx in range(mu.shape[-1]):
                axs.fill_between(
                    data_t[system_idx, _forcasting_traj_idx],
                    np.squeeze(mu[..., state_idx] - 1.96 * sigma[..., state_idx]),
                    np.squeeze(mu[..., state_idx] + 1.96 * sigma[..., state_idx]),
                    color=colors[state_idx],
                    alpha=0.01,
                )

            # markers = ["o", "v", "^", "<", ">", "s", "p", "*"]
            for state_idx in range(mu.shape[-1]):
                axs.plot(
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
            if traj_idx == _forcasting_traj_idx:
                for state_idx in range(mu.shape[-1]):
                    axs.scatter(
                        data_t[system_idx, traj_idx][
                            ctx_mask_with_new_traj_initial_cond_only[
                                system_idx, _forcasting_traj_idx, traj_idx
                            ]
                        ],
                        data_x[system_idx, traj_idx][
                            ctx_mask_with_new_traj_initial_cond_only[
                                system_idx, _forcasting_traj_idx, traj_idx
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
                    axs.plot(
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
                    axs.scatter(
                        data_t[system_idx, traj_idx][
                            ctx_mask_with_new_traj_obs[
                                system_idx, _forcasting_traj_idx, traj_idx
                            ]
                        ],
                        data_x[system_idx, traj_idx][
                            ctx_mask_with_new_traj_obs[
                                system_idx, _forcasting_traj_idx, traj_idx
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

        # Remove the box around the legend
        # legend = plt.legend(frameon=False, fontsize=14)
        axs.set_xlim(t0, t1)
        axs.set_title(kwargs['title'], fontsize=8)
        # plt.tight_layout()
        # plt.savefig(
        #     os.path.join(this_sample_dir, f"{config.model.name}_{config.data.dataset_name}_forcast_prob{config.data.foracsting_problem_prob}_Forcasting_epoch{current_epoch}_on_system{system_idx}.png"),
        #     dpi=300,  # Increase dpi for higher resolution
        # )

def plot_the_param_distribution(**kwargs):
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='np')
    parser.add_argument('--tr_fc_pb', type=float, default=0.5, help='forecasting problem probability')
    args = parser.parse_args()
    plot_parameter_estimation_from_models(seed=0, tr_fc_pb=args.tr_fc_pb)
    # plot_parameter_estimation_from_models(model=args.model, seed=1, tr_fc_pb=args.tr_fc_pb)