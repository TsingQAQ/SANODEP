import os
import ml_collections
from jax import random
from jax import numpy as np
from functools import partial
from NeuralProcesses.objectives import selkov_obj_func, selkov_observer


def distance_penalized_term(x, lower_bound, upper_bound):
    """
    clamp x to remain within lower_bound and upper_bound
    also provide a distance based penalized term to help mitigating the constraint violation
    """
    x_clamped = np.clip(x, lower_bound, upper_bound)
    # use l1 penalized_term
    penalized_term = np.sum(np.abs(x - x_clamped))
    return x_clamped, penalized_term


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    
    # random seed
    config.seed= 0
    
    # model related
    config.model = ml_collections.ConfigDict()
    config.model.name = 'PI_SANODEP_SELKOV'

    config.model.sample_size = 1
    config.model.t0 = 0
    config.model.t1 = 1.0

    # model cfg args
    config.model.cfg_args = ml_collections.ConfigDict()
    config.model.cfg_args.x_dim = 2
    config.model.cfg_args.r_dim = 50
    config.model.cfg_args.encoder_h_dim = 50
    config.model.cfg_args.ode_layer_h_dim = 50
    config.model.cfg_args.decoder_h_dim = 50
    config.model.cfg_args.latent_d_dim = 2
    config.model.cfg_args.latent_l_dim = 2
    config.model.cfg_args.zd_lower_std = 1e-8
    config.model.cfg_args.d_sys_lower_std = 1e-8
    config.model.cfg_args.z0_lower_std = 1e-8
    config.model.cfg_args.tx_to_r_act_fn = 'silu'
    config.model.cfg_args.r_to_z0_act_fn = 'silu'
    config.model.cfg_args.z_to_x_act_fn = 'silu'
    config.model.cfg_args.tx0_to_r_global_act_fn = 'silu'
    config.model.cfg_args.r_to_d_sys_mu_sigma = 'silu'
    config.model.cfg_args.t_embedding = None # 'SEFT'
    config.model.cfg_args.ode_step_size_controller_rtol = 1e-5
    config.model.cfg_args.ode_step_size_controller_atol = 1e-7

    config.model.cfg_args.maximum_timescale = 2 * (config.model.t1 - config.model.t0)
    config.model.cfg_args.time_scaling_coefficient = 10.0
    config.model.cfg_args.param_clamping= partial(distance_penalized_term, 
                                                  lower_bound=np.array([0.0, 0.1]),
                                                    upper_bound=np.array([0.6, 1.1]))

    # model init args
    config.model.init_args = ml_collections.ConfigDict()
    config.model.init_args.t_context = np.zeros(shape=(1, 1))
    config.model.init_args.x_context = np.zeros(shape=(1, 1, 2))
    config.model.init_args.t_target = np.zeros(shape=(1, 1))
    config.model.init_args.x_target = np.zeros(shape=(1, 1, 2))
    config.model.init_args.context_mask = np.ones(shape=(1, 1))
    config.model.init_args.target_mask = np.ones(shape=(1, 1))
    config.model.init_args.ctx_mask_with_new_traj_obs = np.ones(shape=(1, 1, 1))
    config.model.init_args.ctx_mask_with_new_traj_target_mask = np.ones(shape=(1, 1, 1))
    config.model.init_args.target_initial_cond_mask = np.ones(shape=(1, 1))
    config.model.init_args.context_mask_within_known_traj = np.ones(shape=(1, 1))
    config.model.init_args.t0 = config.model.t0 
    config.model.init_args.t1 = config.model.t1
    config.model.init_args.training = True
    config.model.init_args.sample_size = 1
    config.model.init_args.sample_rng = random.PRNGKey(0)
    
    # data related
    config.data = ml_collections.ConfigDict()
    config.data.aux_eval_metric = {'MSE': 'SANODEPTestMSELossAcceptBatchData', 'NLL': 'SANODEPNLLLossAcceptBatchData'} 
    config.data.dataset_name = 'SELKOV'
    config.data.shuffle_buffer_size = 1000
    config.data.num_context_range = (1, 10)
    config.data.use_initial = True # note this is the defualt choice used in NODEP repository, refer TimeNeuralProcessTrainer class
    config.data.num_extra_target_range = (0, 45)
    config.data.known_traj_range = (0, 10)
    config.data.foracsting_problem_prob = 0.5 # 1.0: always do forcasting problem, 0.0: always do regression problem 

    config.data.args = ml_collections.ConfigDict()
    config.data.args.data_gen_rng = random.PRNGKey(0)
    config.data.args.dynamics_smp_num = 20
    config.data.args.initial_condition_smp_num = 100
    config.data.args.num_timesteps = 100
    
    config.data.args.t_range = (0, 1.0)
    config.data.args.x_0_range =  ((0.1, 0.1), (3.0, 3.0))
    config.data.args.a_range = (0.1, 0.5) 
    config.data.args.b_range = (0.25, 0.95) 
    config.data.args.generator = True

    config.data.args.aux = {'MSE': (20, 100), 'NLL': (20, 100)} 
    config.data.eval_metrics = {'MSE': 'SANODEPTestMSELossAcceptBatchData', 'NLL': 'SANODEPNLLLossAcceptBatchData'} 
    # loss
    config.loss_method = 'GreyBoxNeuralODEProcessMFVILossConddsysLoss' 
    config.loss = ml_collections.ConfigDict()
    config.loss.likelihood_reg_weight = 1.0
    config.loss.d_penality_weight = 1.0

    # training related
    config.training = ml_collections.ConfigDict()
    
    config.training.batch_size = 10
    config.training.num_epochs = 310
    config.training.optimizer = ml_collections.ConfigDict()
    config.training.optimizer.name ='rmsprop'
    config.training.optimizer.args = ml_collections.ConfigDict()
    config.training.optimizer.args.num_warmup_epochs = 1
    config.training.optimizer.args.peak_lr = 1e-3 # not exactly sure how to set these hyperparams
    config.training.optimizer.args.initial_lr =  1e-3
    config.training.optimizer.args.learning_rate = 1e-3
    config.training.optimizer.args.end_lr = 1e-3
    config.training.optimizer.args.num_decay_epochs = 2
    config.training.optimizer.args.num_steps_per_epoch = 50 # int(config.data.args.dynamics_smp_num / config.training.batch_size) # batch size 5
    config.training.optimizer.grad_clip = 1e10 # pseudo not enable grad clip
    # config.training.optimizer.args.warmup_steps = config.training.num_epochs * 0.2

    # validation/test
    config.training.snapshot_ckpt_freq = 100

    # intermediate model evaluation sampling
    config.training.eval_snapshot_sampling = True
    config.sampling_fn = sampling_plot
    config.training.snapshot_sampling_freq = 100
    config.snap_shot_sampling_cfg = ml_collections.ConfigDict()
    config.snap_shot_sampling_cfg.dynamics_sample_number = 1
    config.snap_shot_sampling_cfg.model_sample_size = 32
    config.snap_shot_sampling_cfg.sampling_rng = random.PRNGKey(0)
    # evaluation
    config.evaluation = ml_collections.ConfigDict()
    config.evaluation.batch_size = 50
    config.evaluation.num_steps = 1000
    config.evaluation.rng = random.PRNGKey(1000)
    # optimization 
    config.experimental_design = ml_collections.ConfigDict()
    config.experimental_design.rng = random.PRNGKey(0) # this rng will control the problem ODE system, the initial condition
    config.experimental_design.type = 'ActiveLearning'
    config.experimental_design.observer_type = 'Script'
    config.experimental_design.fixed_problem = True
    config.experimental_design.trajectory_aware = True

    # 2D state dim cases
    config.experimental_design.states_num = 2
    config.experimental_design.x0_lower_bound = np.asarray([0.1, 0.1])
    config.experimental_design.x0_upper_bound = np.asarray([0.5, 0.5])
    config.experimental_design.num_traj_iter = 10
    config.experimental_design.t0 = 0.0
    config.experimental_design.t1 = 1.0
    config.experimental_design.time_delay = 0.1001
    config.experimental_design.maximum_obs_per_traj = np.floor((config.experimental_design.t1 - config.experimental_design.t0) / config.experimental_design.time_delay).astype(np.int32)
    
    # initial condition
    config.experimental_design.initial_traj_num = 1
    config.experimental_design.initial_obs_time = np.linspace(0.0, 1.0, config.experimental_design.maximum_obs_per_traj)
    config.experimental_design.batch_size_change_times = np.floor(config.experimental_design.maximum_obs_per_traj / 2).astype(np.int32)
    config.experimental_design.observer = partial(selkov_observer, 
                                                  a_range = config.data.args.a_range, 
                                                  b_range = config.data.args.b_range)
    config.experimental_design.obj_func_form = selkov_obj_func

    config.experimental_design.acq_mc_size = 32

    config.experimental_design.global_maximum_val = 2.7291872295100887
    config.experimental_design.ref_points = np.array([-0.47423223,  5.44026672])

    config.experimental_design.acq_opt = ml_collections.ConfigDict()
    config.experimental_design.acq_opt.initial_smp_num = 50
    config.experimental_design.acq_opt.acq_opt_parallel_num = 10
    config.experimental_design.acq_opt.acq_opt_max_iter = 100
    return config


def sampling_plot(**kwargs):
    """
    Here we make two plot, one is for interpolating, the other one is for forcasting

    We will make use of the auxilary data to make the plot
    """
    from jax import vmap
    from matplotlib import pyplot as plt
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

        # Use a color map
        # colors = cm.get_cmap("tab10")
        # colors = cm.viridis(np.linspace(0, 1, data_x.shape[-1])) 
        colors = cm.viridis(np.linspace(0, 0.9, data_x.shape[-1]))  # Use a subset of the 'viridis' color map
        colors = colors[::-1]  # Reverse the color map
        plt.figure()
        _, axs = plt.subplots()
        model_pred_label = False
        # Use a serif font
        plt.rcParams["font.family"] = "serif"
        axs.tick_params(axis='both', which='major', labelsize=14)  # Increase tick size
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
                        s=80,
                        zorder=1000,
                        edgecolors="w",  # Add white edge
                        linewidths=0.6,  # Adjust the width of the edge
                        label=(
                            "Context data" if not context_label_added else ""
                        ),  # Add label for context data
                    )
                    context_label_added = True
                    axs.plot(
                        data_t[system_idx, traj_idx],
                        data_x[system_idx, traj_idx][..., state_idx],
                        "--",
                        color=colors[state_idx],
                        linewidth=2,
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
                        s=20,
                        color=colors[state_idx],
                        zorder=50,
                        alpha=0.4,
                        edgecolors='k',  # Add white edge
                        linewidths=0.6,  # Adjust the width of the edge
                    )

        # Remove the box around the legend
        legend = plt.legend(frameon=False, fontsize=14)
        axs.set_xlim(t0, t1)
        plt.tight_layout()
        plt.savefig(
            os.path.join(this_sample_dir, f"{config.model.name}_{config.data.dataset_name}_forcast_prob{config.data.foracsting_problem_prob}_Interpolating_epoch{current_epoch}_on_system{system_idx}.png"),
            dpi=300,  # Increase dpi for higher resolution
        )

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
        plt.figure()
        _, axs = plt.subplots()
        model_pred_label = False
        
        axs.tick_params(axis='both', which='major', labelsize=14)  # Increase tick size
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
                        s=80,
                        zorder=1000,
                        edgecolors="w",  # Add white edge
                        linewidths=0.6,  # Adjust the width of the edge
                        label=(
                            "Context data" if not context_label_added else ""
                        ),  # Add label for context data
                    )
                    context_label_added = True
                    axs.plot(
                        data_t[system_idx, traj_idx],
                        data_x[system_idx, traj_idx][..., state_idx],
                        "--",
                        color=colors[state_idx],
                        linewidth=2,
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
                        s=20,
                        color=colors[state_idx],
                        zorder=50,
                        alpha=0.4,
                        edgecolors='k',  # Add white edge
                        linewidths=0.6,  # Adjust the width of the edge
                    )

        # Remove the box around the legend
        legend = plt.legend(frameon=False, fontsize=14)
        axs.set_xlim(t0, t1)
        plt.tight_layout()
        plt.savefig(
            os.path.join(this_sample_dir, f"{config.model.name}_{config.data.dataset_name}_forcast_prob{config.data.foracsting_problem_prob}_Forcasting_epoch{current_epoch}_on_system{system_idx}.png"),
            dpi=300,  # Increase dpi for higher resolution
        )
