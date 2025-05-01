"""
This script aims to plot the model comparison, in a way such that the legend can be unified
"""

import jax
import tensorflow as tf
from jax import numpy as np
from matplotlib import pyplot as plt
from ml_collections import ConfigDict
from NeuralProcesses.models import build
from NeuralProcesses.data import datasets
from ml_collections import ConfigDict
from flax.training import train_state
from flax.training import checkpoints
import matplotlib.gridspec as gridspec
import tensorflow as tf
import logging
import os
import optax


def plot_parameter_estimation_from_models(seed, tr_fc_pb):
    pb_list = ['Brusselator', 'Selkov', 'SIR', 'LV3D', 'SIRD']
    params_name = {
        "Brusselator": [r'$A$', r'$B$'], 
        "Selkov": [r'$a$', r'$b$'],
        "SIR": [r'$\beta$', r'$\gamma$'],
        'LV3D': [r'$\alpha$', r'$\beta$', r'$\epsilon$', r'$\delta$', r'$\gamma$', r'$\zeta$', r'$\eta$', r'$\theta$'],
        'SIRD': [r'$\beta$', r'$\gamma$', r'$\mu$']
    }
    params_attribute_name = {
        "Brusselator": ['A_range', 'B_range'], 
        "Selkov": ['a_range', 'b_range'],
        "SIR": ['beta_range', 'gamma_range'],
        'LV3D': ['alpha_range', 'beta_range', 'epsilon_range', 'delta_range', 'gamma_range', 'zeta_range', 'eta_range', 'theta_range'],
        'SIRD': ['beta_range', 'gamma_range', 'mu_range']
    }
    problem_name_mapping = {
        "LV": "lotka_voterra",
        "Brusselator": "brusselator",
        "FitzHug": "FitzHugh_Nagumo",
        "Selkov": "selkov",
        "SIR": "sir_unnormalized",
        "LV3D": "lotka_voterra_3d",
        "SIRD": "sird"
    }
    problem_title = {
        "LV": "Lotka-Voterra ($2d$)",
        "Brusselator": "Brusselator ($2d$)",
        "FitzHug": "FitzHugh-Nagumo ($2d$)",
        "Selkov": "Selkov ($2d$)",
        "SIR": "SIR  ($3d$)",
        "LV3D": "Lotka-Voterra ($3d$)",
        "SIRD": "SIRD ($4d$)"  
    }

    plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(6, 6)
    import importlib.util

    def load_config_from_py(config_file):
        spec = importlib.util.spec_from_file_location("config", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.get_config()  # Call the get_config function

    current_plot_row = 0
    axs_list = []
    axs_map = {}
    for pb in pb_list:
        # plot the original SANODEP as references
        ax = plt.subplot(gs[current_plot_row, 0])
        # plot the original SANODEP
        config_file = os.path.join("exps/cfgs/", problem_name_mapping[pb], f"sanodep.py")
        config = load_config_from_py(config_file)
        config.data.foracsting_problem_prob = 0.5  # this does not matter that much, as we will do forecasting and interpolating anyhow
        workdir = (
            f"exps/experiments/{problem_name_mapping[pb]}/sanodep/forcast_prob{tr_fc_pb}/seed_{seed}"
            if pb != "LV"
            else f"exps/experiments/{problem_name_mapping[pb]}/model_comparison/sanodep/forcast_prob0.5/seed_{seed}"
        )
        #
        # this works, temperally disable it to increase debugging speed
        axs_list, axs_map, current_plot_row = debug_plot_meta_learn_model(config, workdir, ax, title=problem_title[pb], axs_list=axs_list, axs_map=axs_map, current_plot_row=current_plot_row, gs=gs, params_name=params_name[pb], 
                                                                          params_attribute_name=params_attribute_name[pb], have_axs=True, label=None)

        # Plot the physics enhanced SANODEP
        
        config_file = os.path.join(
            "exps/cfgs/", problem_name_mapping[pb], f"pi_sanodep_clamp.py" # f"grey_box_LV_sanodep.py"
        )
        config = load_config_from_py(config_file)
        config.data.foracsting_problem_prob = 0.5  # this does not matter that much, as we will do forecasting and interpolating anyhow
        workdir = (
            f"exps/experiments/{problem_name_mapping[pb]}/pi_sanodep_clamp/forcast_prob{tr_fc_pb}/seed_{seed}"
            if pb != "LV"
            else f"exps/experiments/{problem_name_mapping[pb]}/model_comparison/pi_sanodep_clamp/forcast_prob0.5/seed_{seed}"
        )
        axs_list, axs_map, _ = debug_plot_meta_learn_model(config, workdir, None, title=problem_title[pb], current_plot_row=current_plot_row, gs=gs, 
                                    params_name=params_name[pb], axs_list=axs_list, axs_map=axs_map, 
                                    params_attribute_name=params_attribute_name[pb], have_axs=False, label='CLIP', plot_model = False)

        config_file = os.path.join(
            "exps/cfgs/", problem_name_mapping[pb], f"pi_sanodep_clamp_le_reg.py" # f"grey_box_LV_sanodep.py"
        )
        config = load_config_from_py(config_file)
        config.data.foracsting_problem_prob = 0.5  # this does not matter that much, as we will do forecasting and interpolating anyhow
        workdir = (
            f"exps/experiments/{problem_name_mapping[pb]}/pi_sanodep_clamp_le_reg/forcast_prob{tr_fc_pb}/seed_{seed}"
            if pb != "LV"
            else f"exps/experiments/{problem_name_mapping[pb]}/model_comparison/pi_sanodep_clamp_le_reg/forcast_prob0.5/seed_{seed}"
        )
        axs_list, axs_map, _ = debug_plot_meta_learn_model(config, workdir, None, title=problem_title[pb], current_plot_row=current_plot_row, gs=gs, 
                                    params_name=params_name[pb], axs_list=axs_list, axs_map=axs_map, 
                                    params_attribute_name=params_attribute_name[pb], have_axs=True, label='CLIP+LE', plot_model = False)
# 
        config_file = os.path.join(
            "exps/cfgs/", problem_name_mapping[pb], f"pi_sanodep_clamp_d_penalty.py" # f"grey_box_LV_sanodep.py"
        )
        config = load_config_from_py(config_file)
        config.data.foracsting_problem_prob = 0.5  # this does not matter that much, as we will do forecasting and interpolating anyhow
        workdir = (
            f"exps/experiments/{problem_name_mapping[pb]}/pi_sanodep_clamp_d_penalty/forcast_prob{tr_fc_pb}/seed_{seed}"
            if pb != "LV"
            else f"exps/experiments/{problem_name_mapping[pb]}/model_comparison/pi_sanodep_clamp_d_penalty/forcast_prob0.5/seed_{seed}"
        )
        axs_list, axs_map, _ = debug_plot_meta_learn_model(config, workdir, None, title=problem_title[pb], current_plot_row=current_plot_row, gs=gs, 
                                    params_name=params_name[pb], axs_list=axs_list, axs_map=axs_map, 
                                    params_attribute_name=params_attribute_name[pb], have_axs=True, label='CLIP+PP', plot_model = False)
        
        config_file = os.path.join(
            "exps/cfgs/", problem_name_mapping[pb], f"pi_sanodep_clamp_le_reg_d_penalty.py" # f"grey_box_LV_sanodep.py"
        )
        config = load_config_from_py(config_file)
        config.data.foracsting_problem_prob = 0.5  # this does not matter that much, as we will do forecasting and interpolating anyhow
        workdir = (
            f"exps/experiments/{problem_name_mapping[pb]}/pi_sanodep_clamp_le_reg_d_penalty/forcast_prob{tr_fc_pb}/seed_{seed}"
            if pb != "LV"
            else f"exps/experiments/{problem_name_mapping[pb]}/model_comparison/pi_sanodep_clamp_le_reg_d_penalty/forcast_prob0.5/seed_{seed}"
        )
        axs_list, axs_map, current_plot_row = debug_plot_meta_learn_model(config, workdir, None, title=problem_title[pb], current_plot_row=current_plot_row, gs=gs, 
                                    params_name=params_name[pb], axs_list=axs_list, axs_map=axs_map, 
                                    params_attribute_name=params_attribute_name[pb], have_axs=True, label='CLIP+PP+LE', plot_model = False)
        

    # Plot the parameter estimation result

    # legend = plt.legend(frameon=False, fontsize=14)
    # So far, nothing special except the managed prop_cycle. Now the trick:
    lines_labels = [ax.get_legend_handles_labels() for ax in axs_list]
    # lines_labels = [ax.get_legend_handles_labels() for ax in axs.flat]
    lines, labels = zip(*lines_labels)
    # Combine lines and labels into pairs
    pairs = list(zip(lines, labels))
    # Initialize an empty dictionary
    unique_dict = {}

    for lines, labels in pairs:
        for _line, _label in zip(lines, labels):
            # If the label is not in the dictionary, add the line and label to the dictionary
            if _label != []:
                # print(_label)
                if _label not in unique_dict:
                    unique_dict[_label] = _line

    # Convert the dictionary back to a list of tuples
    unique_pairs = list(unique_dict.items())

    # Unpack unique lines and labels
    unique_labels, unique_lines = zip(*unique_pairs)

    # Create legend with unique lines and labels
    # axs.legend(unique_labels, unique_lines, loc='lower center')
    # Get the current figure
    fig = plt.gcf()

    # Create legend with unique lines and labels for the entire figure
    # Create legend with unique lines and labels for the entire figure
    # fig.legend(unique_lines, unique_labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=9, fontsize=6, markerscale=2)

    # Adjust the subplots to create space for the legend
    plt.subplots_adjust(bottom=0.1)  # Adjust the bottom parameter to create more space

    # fig = plt.gcf()  # Get the current figure
    fig.legend(
        unique_lines,
        unique_labels,
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    # Finally, the legend (that maybe you'll customize differently)
    # plt.legend(lines_dict.values(), lines_dict.keys(), loc='lower center', ncol=4)
    # axs.set_xlim(config.model.t0, config.model.t1)
    # plt.tight_layout(pad=0.9, w_pad=0.5, h_pad=0.1)
    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
    # plt.tight_layout()
    # axs_list[0].set_ylabel("Value", fontsize=5)
    # axs_list[2].set_ylabel("Density", fontsize=5)
    # axs[0].set_ylim(0, 5)
    # axs[1].set_ylim(0, 5)
    plt.subplots_adjust(
        left=0.05, right=0.99, bottom=0.1, top=0.95, wspace=0.2, hspace=0.48
    )
    plt.savefig(
        "param_est_debug.png",
        dpi=500,  # Increase dpi for higher resolution
    )


def debug_plot_meta_learn_model(config: ConfigDict, workdir: str, axs, title, **kwargs):
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    assert isinstance(config.seed, int)  # type check
    rng = jax.random.PRNGKey(config.seed)
    if 'PI_SANODEP' in config.model.name:
        config.model.cfg_args.return_param_est = True
    rng, init_rng = jax.random.split(rng)
    model, init_model_state, initial_params, rng = build.init_model(
        init_rng, config
    )
    # instantiate optimizers

    # as some model is trained with gradient clip and replace nan but the others does not, 
    # this have imposed some compatibility of load model checkpoint, here as a tmp workaround we try both to load the model
    # create training state
    # if "create_optimizer" in config.training:
    #     optimizer = config.training["create_optimizer"](config)
    # else:
    #     optimizer = create_optimizer(config)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    try:
        optimizer = optax.rmsprop(learning_rate=config.training.optimizer.args.learning_rate)
        training_state = train_state.TrainState.create(
            apply_fn=model.apply, params=initial_params, tx=optimizer
        )
        # Resume training when intermediate checkpoints are detected
        training_state = checkpoints.restore_checkpoint(checkpoint_dir, training_state)
    except:
        optimizer = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(1.0), optax.rmsprop(learning_rate=config.training.optimizer.args.learning_rate))
        training_state = train_state.TrainState.create(
            apply_fn=model.apply, params=initial_params, tx=optimizer
        )
        # Resume training when intermediate checkpoints are detected
        training_state = checkpoints.restore_checkpoint(checkpoint_dir, training_state)
    # `state.step` is JAX integer on the GPU/TPU devices
    initial_step = int(training_state.step)
    trained_epoch = (
        initial_step // config.training.optimizer.args.num_steps_per_epoch
    )
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

    this_sample_dir = sample_dir  # os.path.join(, "epoch_{}".format(current_epoch))
    tf.io.gfile.makedirs(this_sample_dir)
    axis_list, axs_map, current_plot_row = meta_learn_model_interploating_plot(
                                            config=config,
                                            model=model,
                                            rng=rng,
                                            training_state=training_state,
                                            dataset_inst=dataset_inst,
                                            current_epoch=trained_epoch,
                                            aux_batch=aux_datsets,
                                            this_sample_dir=this_sample_dir,
                                            plot_axs=axs,
                                            title=title,
                                            current_plot_row = kwargs["current_plot_row"],
                                            gs = kwargs['gs'],
                                            params_name = kwargs['params_name'],\
                                            axs_map =  kwargs["axs_map"],
                                            axs_list=kwargs["axs_list"], 
                                            params_attribute_name=kwargs["params_attribute_name"], 
                                            have_axs=kwargs["have_axs"], 
                                            label=kwargs["label"]
                                            )
    return axis_list, axs_map, current_plot_row


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
    data_t, data_x, data_params = kwargs["aux_batch"]["NLL"]
    rng = kwargs["rng"]
    model = kwargs["model"]
    current_plot_row = kwargs["current_plot_row"]
    sample_system_number = config.snap_shot_sampling_cfg.dynamics_sample_number
    traj_size = data_x.shape[1]
    training_state = kwargs["training_state"]
    dataset_inst = kwargs["dataset_inst"]
    axs_list = kwargs["axs_list"]
    axs_map = kwargs["axs_map"]
    t0 = config.model.t0
    t1 = config.model.t1
    # why not directly make use of data_preprocessor?
    pre_processor = get_data_preprocessor(dataset_inst, config)
    # pre_processor(dataset_inst, config)
    aux_data_batch = (data_t, data_x, data_params)
    try:
        processed_data, rng = pre_processor(
            aux_data_batch, rng, known_traj_range=config.data.known_traj_range
        )
    except:
        processed_data, rng = pre_processor(
            (data_t, data_x, data_params), rng, known_traj_range=config.data.known_traj_range
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
        if kwargs["config"].model.name == "SANODEP":
            x_pred_f, x_pred_sigma = vmap(batch_model_apply)(
                data_t,
                data_x,
                data_t,
                data_x,
                target_initial_cond_mask,
                ctx_mask_with_new_traj_obs,
                ctx_mask_with_new_traj_target_mask,
            )  # [batch_size, traj_size, num_samples, num_points, output_dim], [batch_size, traj_size, num_samples, num_points, output_dim]
        else: # PI-SANODEP
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
        colors = cm.viridis(
            np.linspace(0, 0.9, 10) # 5)
        )  # cm.viridis(np.linspace(0, 0.9, data_x.shape[-1]))  # Use a subset of the 'viridis' color map
        colors = colors[::-1]  # Reverse the color map
        all_axes = kwargs["plot_axs"]
        model_pred_label = False
        # Use a serif font
        plt.rcParams["font.family"] = "serif"
        # all_axes.tick_params(axis='both', which='major', labelsize=6)  # Increase tick size
        if kwargs["config"].model.name == "SANODEP":
            axs = all_axes
            axs.set_title(kwargs["title"]+ "\n SANODEP-$\lambda = 0.5$", fontsize=8, pad=3)
            axs.set_ylabel("Value", fontsize=7)
        else:
            axs = plt.subplot(kwargs['gs'][kwargs["current_plot_row"], 1])
            axs.set_title(kwargs["title"]+ "\n PI-SANODEP-$\lambda = 0.5$", fontsize=8, pad=3)
            # axs = all_axes[1]
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
                    linewidth=0.2,
                    label=(
                        f"Predicted State {state_idx + 1} Value"
                        if not model_pred_label
                        else ""
                    ),
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
                        color="k",
                        s=15,
                        zorder=1000,
                        edgecolors="w",  # Add white edge
                        linewidths=0.2,  # Adjust the width of the edge
                        label=(
                            r"$\mathcal{T}_{new}^{\mathbb{C}}$"
                            if not new_context_label_added
                            else ""
                        ),  # Add label for context data
                    )
                    new_context_label_added = True
                    axs.plot(
                        data_t[system_idx, traj_idx],
                        data_x[system_idx, traj_idx][..., state_idx],
                        "--",
                        color=colors[state_idx],
                        linewidth=1.0,
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
                        edgecolors="k",  # Add white edge
                        linewidths=0.4,  # Adjust the width of the edge
                        label=(
                            r"$\mathcal{T}^{\mathbb{C}}$"
                            if not context_label_added
                            else ""
                        ),  # Add label for context data
                    )
                context_label_added = True
        # set x and y tick font size
        
        axs.tick_params(axis="both", which="major", labelsize=6)
        # Remove the box around the legend
        # legend = plt.legend(frameon=False, fontsize=14)
        axs.set_xlim(t0, t1)
        axs_list.append(axs)
        axs_map['0,0']=axs
        # PI-SANODEP
        if kwargs["config"].model.name != "SANODEP":
            param_est_mean, param_est_std = params_est
            params_num = param_est_mean.shape[-1]
            for param_idx in range(params_num):
                # if the 2 + param_idx > 6 then we plot it in a new line
                # print(axs_map.keys())
                if 2 + param_idx < 6:
                    if not kwargs['have_axs']:
                        axs = plt.subplot(kwargs['gs'][current_plot_row, 2 + param_idx])
                        axs_map[f'{current_plot_row},{2 + param_idx}'] = axs
                    else:
                        axs = axs_map[f'{current_plot_row},{2 + param_idx}']
                else:
                    if not kwargs['have_axs']:
                        axs = plt.subplot(kwargs['gs'][current_plot_row + 1, (2 + param_idx) % 6])
                        axs_map[f'{current_plot_row + 1},{(2 + param_idx) % 6}'] = axs
                    else:
                        axs = axs_map[f'{current_plot_row + 1},{(2 + param_idx) % 6}']
                if param_idx == 0:
                    axs.set_ylabel("Density", fontsize=7)
                from scipy.stats import lognorm

                scale_alpha = np.exp(param_est_mean[system_idx, _interpolating_traj_idx, param_idx])
                # s_alpha = np.sqrt(param_est_std[system_idx, _interpolating_traj_idx, param_idx])
                s_alpha = param_est_std[system_idx, _interpolating_traj_idx, param_idx]
                # Generate x values
                x_alpha = np.linspace(lognorm.ppf(0.01, s_alpha, scale=scale_alpha),
                     lognorm.ppf(0.99, s_alpha, scale=scale_alpha), 100)
                # np.linspace(0, 2, 100  ) 
                # Generate y values (lognormal PDF)
                y_alpha = lognorm.pdf(x_alpha, s_alpha, scale=scale_alpha)
                axs.plot(x_alpha, y_alpha, label=kwargs['label'])
                ylims = axs.get_ylim()
                axs.vlines(
                    data_params[system_idx, traj_idx, param_idx],
                    ylims[0],
                    ylims[1],
                    color="r",
                    linestyle="--",
                    linewidth=0.8
                )
                axs.set_yticks([])
                axs.set_title(kwargs["params_name"][param_idx], fontsize=10)
                # axs.set_xlim(0, 2)
                # axs.set_xlim(x_alpha[0], x_alpha[-1])
                param_range = getattr(config.data.args, kwargs["params_attribute_name"][param_idx])[1] - getattr(config.data.args, kwargs["params_attribute_name"][param_idx])[0]
                # sampled_params[system_idx, traj_idx, param_idx] - 0.25 * param_range
                axs.set_xlim(0.0, 
                             max(data_params[system_idx, traj_idx, param_idx] + 0.25 * param_range, lognorm.ppf(0.95, s_alpha, scale=scale_alpha)))
                axs.tick_params(axis="both", which="major", labelsize=6)
                axs_list.append(axs)
        if kwargs["config"].model.name == "SANODEP":
            return axs_list, axs_map, current_plot_row
        else:
            return axs_list, axs_map, current_plot_row + 1 if 2 + param_idx < 6 else current_plot_row + 2
            # (
            #     mu_alpha,
            #     sigma_alpha,
            #     mu_beta,
            #     sigma_beta,
            #     mu_gamma,
            #     sigma_gamma,
            #     mu_delta,
            #     sigma_delta,
            # ) = params_est
            # mu_alpha = mu_alpha[system_idx, traj_idx]
            # sigma_alpha = sigma_alpha[system_idx, traj_idx]
            # mu_beta = mu_beta[system_idx, traj_idx]
            # sigma_beta = sigma_beta[system_idx, traj_idx]
            # mu_gamma = mu_gamma[system_idx, traj_idx]
            # sigma_gamma = sigma_gamma[system_idx, traj_idx]
            # mu_delta = mu_delta[system_idx, traj_idx]
            # sigma_delta = sigma_delta[system_idx, traj_idx]
            # from scipy.stats import lognorm
# 
            # scale_alpha = np.exp(mu_alpha)
            # s_alpha = np.sqrt(sigma_alpha)
# 
            # # Generate x values
            # x_alpha = np.linspace(
            #     0, 2, 100
            # )  # np.linspace(lognorm.ppf(0.001, s_alpha, scale=scale_alpha),
            # #       lognorm.ppf(0.999, s_alpha, scale=scale_alpha), 100)
# 
            # # Generate y values (lognormal PDF)
            # y_alpha = lognorm.pdf(x_alpha, s_alpha, scale=scale_alpha)
            # all_axes[2].plot(x_alpha, y_alpha)
            # ylims = all_axes[2].get_ylim()
            # all_axes[2].vlines(
            #     alpha[system_idx, traj_idx],
            #     ylims[0],
            #     ylims[1],
            #     color="r",
            #     linestyle="--",
            #     linewidth=0.8,
            # )
            # all_axes[2].set_yticks([])
            # all_axes[2].set_title(r"$\alpha$")
            # all_axes[2].set_xlim(0, 2)
# 
            # scale_beta = np.exp(mu_beta)
            # s_beta = np.sqrt(sigma_beta)
            # x_beta = np.linspace(
            #     0, 2, 100
            # )  # np.linspace(lognorm.ppf(0.001, s_beta, scale=scale_beta),
            # #        lognorm.ppf(0.999, s_beta, scale=scale_beta), 100)
# 
            # # Generate y values (lognormal PDF)
            # y_beta = lognorm.pdf(x_beta, s_beta, scale=scale_beta)
            # all_axes[3].plot(x_beta, y_beta)
            # ylims = all_axes[3].get_ylim()
            # all_axes[3].vlines(
            #     beta[system_idx, traj_idx],
            #     ylims[0],
            #     ylims[1],
            #     color="r",
            #     linestyle="--",
            #     linewidth=0.8,
            # )
            # all_axes[3].set_yticks([])
            # all_axes[3].set_title(r"$\beta$")
            # all_axes[3].set_xlim(0, 2)
# 
            # scale_delta = np.exp(mu_delta)
            # s_delta = np.sqrt(sigma_delta)
            # x_delta = np.linspace(
            #     0, 2, 100
            # )  # np.linspace(lognorm.ppf(0.001, s_delta, scale=scale_delta),
            # #       lognorm.ppf(0.999, s_delta, scale=scale_delta), 100)
# 
            # # Generate y values (lognormal PDF)
            # y_delta = lognorm.pdf(x_delta, s_delta, scale=scale_delta)
            # all_axes[4].plot(x_delta, y_delta)
            # ylims = all_axes[4].get_ylim()
            # all_axes[4].vlines(
            #     delta[system_idx, traj_idx],
            #     ylims[0],
            #     ylims[1],
            #     color="r",
            #     linestyle="--",
            #     linewidth=0.8,
            # )
            # all_axes[4].set_yticks([])
            # all_axes[4].set_title(r"$\delta$")
            # all_axes[4].set_xlim(0, 2)
# 
            # scale_gamma = np.exp(mu_gamma)
            # s_gamma = np.sqrt(sigma_gamma)
            # # x_gamma = np.linspace(lognorm.ppf(0.001, s_gamma, scale=scale_gamma),
            # #                 lognorm.ppf(0.999, s_gamma, scale=scale_gamma), 100)
            # x_gamma = np.linspace(0, 2, 100)
# 
            # # Generate y values (lognormal PDF)
            # y_gamma = lognorm.pdf(x_gamma, s_gamma, scale=scale_gamma)
            # all_axes[5].plot(x_gamma, y_gamma)
            # ylims = all_axes[5].get_ylim()
            # all_axes[5].vlines(
            #     gamma[system_idx, traj_idx],
            #     ylims[0],
            #     ylims[1],
            #     color="r",
            #     linestyle="--",
            #     linewidth=0.8,
            # )
            # all_axes[5].set_yticks([])
            # all_axes[5].set_title(r"$\gamma$")
            # all_axes[5].set_xlim(0, 2)

            # get the ground truth
        # plt.tight_layout()
        # plt.savefig(
        #     os.path.join(this_sample_dir, f"{config.model.name}_{config.data.dataset_name}_forcast_prob{config.data.foracsting_problem_prob}_Interpolating_epoch{current_epoch}_on_system{system_idx}.png"),
        #     dpi=300,  # Increase dpi for higher resolution
        # )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='np')
    parser.add_argument(
        "--tr_fc_pb", type=float, default=0.5, help="forecasting problem probability"
    )
    args = parser.parse_args()
    plot_parameter_estimation_from_models(seed=0, tr_fc_pb=args.tr_fc_pb)
    # plot_parameter_estimation_from_models(model=args.model, seed=1, tr_fc_pb=args.tr_fc_pb)
