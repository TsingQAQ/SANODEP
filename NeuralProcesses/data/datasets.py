"""
The training datasets module used in Neural Process for meta learning of dynamic systems

Note: for the termination time of the dynamical system, we follow the approach of NODEP's example on Lotka Voterra, 
which have scaled the time range from [0, 15] to [0, 1.5], I am not exactly sure why this is done, 
but we hope to follow the same approach here by introducing this additional time scaling factor

refer here is also a discussion on the termination time, though specifically when using Neural ODEs to model dynamical systems, 
the opinion on whether to scale the final time remains open
https://www.reddit.com/r/MachineLearning/comments/giqy79/d_role_of_final_time_t_in_neural_odes/
"""

from abc import ABC, abstractclassmethod
from jax import numpy as np
from jax import random
import tensorflow as tf
from gpjax.mean_functions import Zero
from gpjax import kernels as jk
from gpjax.gps import Prior
from jax import vmap
from jax import jit
from jax.random import normal, uniform, randint
from ml_collections import ConfigDict
from typing import Optional, List
from .utils import (
    context_target_mask_gen_dynamic_system,
)
from diffrax import diffeqsolve, PIDController, ODETerm, SaveAt, Dopri5
from jax.typing import ArrayLike
from typing import Optional
from einops import rearrange
from functools import partial
from trieste.data import Dataset
import diffrax
import sys


thismodule = sys.modules[__name__]


class TFDataset(ABC):
    """
    Dataset prepared in a tensorflow dataset manner
    """

    def __init__(self):
        super().__init__()

    @abstractclassmethod
    def generate_tf_dataset(self):
        """
        Generate tensorflow type dataset

        Note the output MUST be 2D even it is single output dataset
        """

    def get_data_input_scaler(self):
        """
        Get data input scaler
        """

    def get_data_inverse_scaler(self):
        """
        Get data inverse scaler
        """

    def get_aux_datasets(self):
        """
        Get auxiliary datasets, e.g. validation or test datasets
        """

def get_dataset(config):
    """
    Interface for train_evel.py
    """
    data_inst = getattr(thismodule, config.data.dataset_name)()
    train_data = data_inst.generate_tf_dataset(**config.data.args)
    return train_data, data_inst


def get_data_preprocessor(data_inst, config: ConfigDict):
    if isinstance(
        data_inst,
        (SIRD, LOTKA_VOLTERRA_ODE,  Brusselator, SELKOV, LOTKA_VOLTERRA_ODE_3D, gp_ode_prior, SIR_Unormalized_ODE),
    ):
        @partial(jit, static_argnames=("all_as_target"))
        def data_pre_processor(
            data_batch: tuple[ArrayLike],
            rng: random.PRNGKey,
            known_traj_range: tuple, 
            all_as_target: bool = False,
        ):
            """
            :param all_as_target in testing, all the  data is actually used as the target data
                this is utilized to mostly reproduce the same result in the neural ode process paper
            :param known_traj_range the range of context trajectories within a dynamical system
            :param data_batch: a tuple of (data_x, data_y)
            """
            data_x, data_y, data_params = data_batch
            rng, step_rng, uniform_rng, uniform_rng2, uniform_rng3 = random.split(
                rng, 5
            )
            batch_system_num, batch_traj_size, num_points = (
                data_x.shape[0],
                data_x.shape[1],
                data_x.shape[2],
            )
            # [system_num, traj_size] how many context observations within trajectories
            num_context = randint( # [system_num, traj_size] how many context observations within trajectories
                uniform_rng,
                (
                    batch_system_num,
                    batch_traj_size,
                ),
                *config.data.num_context_range,
            )
            if all_as_target:  # use all the rest as target
                num_extra_target = data_x.shape[2] - num_context
            else:
                num_extra_target = randint( # [system_num, traj_size] how many extra target observations within trajectories
                    uniform_rng2,
                    (
                        batch_system_num,
                        batch_traj_size,
                    ),
                    *config.data.num_extra_target_range,
                )
            # [system_num] how many trajectories know in each dynamic system
            known_trajectory = randint(
                uniform_rng3, (batch_system_num,), *known_traj_range
            )
            (
                context_mask,
                target_mask,
                ctx_mask_existing_known_traj,
                ctx_mask_with_new_traj_obs,
                ctx_mask_with_new_traj_target_mask,
                target_initial_cond_mask,
                target_mask_unknown_traj,
            ) = context_target_mask_gen_dynamic_system(
                step_rng,
                batch_size=batch_system_num,
                known_traj=known_trajectory,
                num_timesteps=num_points,
                num_context=num_context,
                num_extra_target=num_extra_target,
                problem_setting_forcasting_prob=config.data.foracsting_problem_prob,
                know_initial=config.data.get("use_initial", False),
            )
            data_tuples = (
                data_x,
                data_y,
                data_params, 
                context_mask,
                target_mask,
                ctx_mask_with_new_traj_obs,
                ctx_mask_with_new_traj_target_mask,
                target_initial_cond_mask,
                target_mask_unknown_traj,
                known_trajectory,
            )
            return data_tuples, rng

        return data_pre_processor
    else:
        # mainly used for generator based data
        return lambda datas, rng, **kwargs: (
            (*data_inst.get_data_input_scaler(datas[:2]), *datas[2:]),
            rng,
        )  # use the forward 2 as the rest are mask


def get_data_post_processor(data_inst, config: ConfigDict):
    return data_inst.get_data_inverse_scaler


def get_aux_datasets(dataset_inst, config: ConfigDict):
    if isinstance(
        dataset_inst,
        (
            gp_ode_prior, 
            LOTKA_VOLTERRA_ODE,
            Brusselator, 
            SELKOV, 
            SIRD,
            LOTKA_VOLTERRA_ODE_3D, 
            SIR_Unormalized_ODE, 
            SIRD
        ),
    ):
        return dataset_inst.get_aux_datasets()
    else:
        raise NotImplementedError


class gp_ode_prior:
    """
    Generate arbitary time series using GP prior
    dx/dt = f(x, t), where f is a GP prior

    Here we use the random fourier features to generate the priors

    Note that if we use 1-d GP, the thing is
    """

    def generate_tf_dataset(
        self,
        data_gen_rng: random.PRNGKey,
        x_0_range: List,
        lengthscale_range: float | List,
        signal_std_range: float | List,
        t_range: Optional[List] = [0, 1.5],
        num_context_range: Optional[List] = None,
        num_timesteps: Optional[int] = None,
        num_target_range: Optional[int] = None,
        aux: Optional[dict] = None,
        generator: bool = True,
        dynamics_smp_num: int = 20,
        initial_condition_smp_num: int = 20,
        num_train_samples: Optional[int] = np.inf,
        batch_size: Optional[int] = np.inf,
        generator_use_initial: bool = True,
        likelihood_std: float = 0.0,
        time_scaling_coefficient: float = 1.0,
    ) -> tf.data.Dataset:
        """
        :params order_range: since model the r.h.s of dx/dt = f(x) using GP prior typically results in first order ode,
            we may need to carefully think how to make use of this formalism to model higher order ODEs (e.g., oscillator),

        For the 1st order cases,
        since GP prior can have f(x) = 0 and f'(x) < 0, this is known as a stable equillibirum
        and if f(x) = 0 and f'(x) > 0, this is known as unstable equillibirum
        """
        # data_gen_rng = random.PRNGKey(117)
        x_dim = len(x_0_range[0])
        def batch_training_data_generator(rng, batch_size):
            while True:
                t_samples, x_samples, rng, params = self.sample_gp_ode_prior(
                        rng,
                        dynamics_smp_num,
                        initial_condition_smp_num, 
                        num_timesteps, 
                        x_0_range, 
                        t_range,
                        x_dim,
                        lengthscale_range,
                        signal_std_range,
                        likelihood_std, 
                        time_scaling_coefficient)

                yield t_samples, x_samples, params  # [..., N, 1]

        data_gen = lambda: batch_training_data_generator(
            data_gen_rng,
            lengthscale_range,
            signal_std_range,
            x_dim,
            t_range,
            dynamics_smp_num,
            num_context_range,
            likelihood_std,
        )

        # note that this will make all the generated aux datasets the same since we do not change the rng,
        # this is ideal if we want to compare between different models as no need to store same data locally
        if aux is not None:
            data_gen_rng, aux_data_gen_rng = random.split(data_gen_rng, 2)
            aux_datasets = {}
            for key, (num_aug_dynamic, num_aug_initial_cond) in aux.items():
                x_samples, y_samples, _, params = self.sample_gp_ode_prior(
                        aux_data_gen_rng,
                        num_aug_dynamic,
                        num_aug_initial_cond, 
                        num_timesteps, 
                        x_0_range, 
                        t_range,
                        x_dim,
                        lengthscale_range,
                        signal_std_range,
                        likelihood_std,
                        time_scaling_coefficient
                    )
                aux_datasets[key] = (x_samples, y_samples, params)
        else:
            aux_datasets = {}

        self.aux_datasets = aux_datasets

        # only plot usage
        x_samples, y_samples, _, fig, axs = self.sample_gp_ode_prior_for_plot(
                aux_data_gen_rng,
                num_aug_dynamic,
                num_aug_initial_cond, 
                num_timesteps, 
                x_0_range, 
                t_range,
                x_dim,
                lengthscale_range,
                signal_std_range,
                likelihood_std,
                time_scaling_coefficient
            )
        self.plot_datasets = (x_samples, y_samples)
        self.plot_handles = (fig, axs)

        if generator:
            data_gen = lambda: batch_training_data_generator(data_gen_rng, batch_size)
            train_data = tf.data.Dataset.from_generator(
                data_gen, (np.float32, np.float32, np.float32)
            )  # # , np.bool_, np.bool_))
            return train_data
        # train_data = tf.data.Dataset.from_generator(
        #     data_gen, (np.float32, np.float32, np.bool_, np.bool_)
        # )
        # return train_data

    @staticmethod
    @partial(
        jit,
        static_argnames=(
            "num_timesteps",
            "dynamic_sample_num", 
            "initial_cond_sample_num", 
            "initial_condition_range",
            "t_range",
            "x_dim",
            "time_scaling_coefficient", 
            "lengthscale_range",
            "signal_std_range",
            "likelihood_std", 
            "num_context_range",
            "num_target_range",
            "use_initial",
            "specified_times",
        ),
    )
    def sample_gp_ode_prior(
        rng: random.PRNGKey,
        dynamic_sample_num: int,
        initial_cond_sample_num: int,
        num_timesteps: int,
        initial_condition_range: List,
        t_range: List,
        x_dim,
        lengthscale_range,
        signal_std_range,
        likelihood_std,
        time_scaling_coefficient: float = 1.0,
        num_context_range: Optional[List] = None,
        num_target_range: Optional[List] = None,
        use_initial: bool = False,
        specified_times: Optional[ArrayLike] = None,
        solver = 'Dopri5',
    ):
        # rng, sample_rng = random.split(rng)

        # follow Nueral Process implementation, we explicitly split here to context data and target data
        # x_samples = uniform(sample_rng, shape=(batch_size, joint_batch_size, dim), minval=-2.0, maxval=2.0)
        # rng, ctx_rng = random.split(rng)
        # rng, trg_rng = random.split(rng)
        # num_context = randint(
        #     ctx_rng,
        #     shape=(1,),
        #     minval=num_context_range[0],
        #     maxval=num_context_range[1],
        # )[0]
        # if num_all_data_points is not None:  # the total size has been specified
        #     assert not num_target_range, ValueError(
        #         "num_target-range and num_all_data_points cannot be specified simultaneously!"
        #     )
        #     num_extra_target = num_all_data_points - num_context
        # else:
        #     num_extra_target = randint(
        #         trg_rng,
        #         shape=(1,),
        #         minval=num_target_range[0],
        #         maxval=num_target_range[1],
        #     )[0]
        #     num_all_data_points = num_context + num_extra_target

        rng, lengthscale_rng = random.split(rng)
        lengthscales = uniform(  # only support 1d right now
            lengthscale_rng,
            shape=(x_dim,),
            minval=np.asarray(lengthscale_range[0]),
            maxval=np.asarray(lengthscale_range[1]),
        )
        # assert len(signal_std_range) == 2
        rng, signal_std_rng = random.split(rng)
        signal_std = uniform(  # only support 1d right now
            signal_std_rng,
            shape=(1,),
            minval=signal_std_range[0],
            maxval=signal_std_range[1],
        )[0]
        signal_variance = signal_std**2

        sample_fns = [] 
        for _ in range(x_dim):
            kernel = jk.RBF(lengthscale=lengthscales, variance=signal_variance, active_dims=list(range(x_dim)))
            meanf = Zero()
            prior = Prior(mean_function=meanf, kernel=kernel)
            rng, gp_sample_rng = random.split(rng)
            sample_fn = prior.sample_approx(dynamic_sample_num, key=gp_sample_rng, num_features=512)
            sample_fns.append(sample_fn)

        # ode_fn = lambda t, x, args: sample_fn(x)
        @jit
        def dynamics(t, x, args):
            """
            :params x: [num_input, x_dim]
            """
            sample_id = args
            dxs = []
            for i in range(x_dim):
                dx = sample_fns[i](x)[..., sample_id]
                # we add a clip here to avoid the nan/inf, this is for numerical stability only
                # if it is numerically stable, this dx should be almost cover 100% sample space
                # dx = np.clip(dx, -10 * signal_std, 10 * signal_std)
                dxs.append(dx)
            return np.stack(dxs, axis=-1)
            # samples = sample_fn(x)
            # # return  # [num_input, num_samples]
            # return samples[..., sample_id]

        # sample initial state
        rng, x0_rng = random.split(rng)
        sample_x0 = random.uniform(
            x0_rng, 
            minval=np.asarray(initial_condition_range[0]), 
            maxval=np.asarray(initial_condition_range[1]), 
            shape=(dynamic_sample_num, initial_cond_sample_num, x_dim)
        )  # note the last dim is used for rff sample, bit hacky

        if specified_times is None:
            t_samples = np.linspace(t_range[0], t_range[1], num_timesteps)
        else:
            t_samples = specified_times
        raw_t_samples = (
            t_samples * time_scaling_coefficient
        )  # note this is used in NODEP paper that it scaled the time by 10
        raw_t_range = [t_range[0], time_scaling_coefficient * (t_range[1] - t_range[0])]

        # solve ode
        batch_diff_solve = lambda x0, ts, smp_idx: diffeqsolve(
            ODETerm(dynamics),
            t0=raw_t_range[0],
            t1=raw_t_range[1],
            dt0=None,
            solver=getattr(diffrax, solver)(),
            stepsize_controller=PIDController(rtol=1e-5, atol=1.e-5),
            max_steps=10000,
            throw=True,
            y0=x0,
            saveat=SaveAt(ts=raw_t_samples),
            args=smp_idx,
        )

        # batch_diff_solve(sample_x0[0], np.sort(t_samples[0], axis=0)[:, 0], 0)
        f_samples = vmap(batch_diff_solve, (0, None, 0))(
            sample_x0, t_samples, np.arange(dynamic_sample_num)
        ).ys
        # if not np.all(np.isfinite(f_samples)):
        #     # print where the nan/inf is
        #     print("nan/inf detected in the samples")
            # print(np.where(np.isfinite(f_samples)))
            # a = 2
        f_samples = rearrange(f_samples, "dynamics timestep traj_size state_dim -> dynamics traj_size timestep state_dim")
        t_samples = np.repeat(
            np.repeat(t_samples[None, ...], repeats=initial_cond_sample_num, axis=0)[
                None, ...
            ],
            repeats=dynamic_sample_num,
            axis=0,
        )[
            ..., None
        ]  # [dynamic_sample_num, initial_cond_sample_num, timesteps, 1]
        # dynamics_illustration = vmap(debug_ode_fn, (0, 0, 0))(sample_x0, t_samples, np.arange(batch_size))
        # observation noise
        rng, x0_rng = random.split(rng)
        y_samples = f_samples + likelihood_std * normal(
            x0_rng, shape=f_samples.shape
        )
        # from matplotlib import pyplot as plt
        # # # # # Create a grid of points to evaluate the vector field at
        # plt.figure()
        # x1 = np.linspace(-1, 1, 50)
        # x2 = np.linspace(-1, 1, 50)
        # X1, X2 = np.meshgrid(x1, x2)
        # X = np.repeat(np.stack((X1.flatten(), X2.flatten()), axis=-1)[None, ...], dynamic_sample_num, axis=0)
        # DX = vmap(dynamics, in_axes=(None, 0, 0))(None, X, np.arange(dynamic_sample_num))
        # DX1, DX2 = DX[..., 0].reshape(dynamic_sample_num, 50, 50), DX[..., 1].reshape(dynamic_sample_num, 50, 50)
        # # Get the number of batches
        # num_batches = dynamic_sample_num
        # # Get the number of subplots to plot
        # num_subplots = 6 # min(num_batches, 20)
        # # Create a figure with a 2-column grid of subplots
        # 
        # # fig, axs = plt.subplots(num_subplots, 4, figsize=(12, 4*num_subplots))
        # fig, axs =  plt.subplots(4, 6, figsize=(10, 5)) # plt.subplots(num_subplots, 4, figsize=(2, 4*num_subplots))
        # for i in range(num_subplots):
        #     # Get the trajectory and time samples for this batch
        #     traj_i = y_samples[i, 0, :]  # Assuming the 1st trajectory is what you want
        #     t_samples_i = t_samples[i, 0, :, 0]  # Assuming the 1st initial condition
        #     # Plot the contour of DX1 in the second column
        #     axs[0, i].contourf(X1, X2, DX1[i], cmap='viridis')
        #     axs[0, i].set_title(f'Batch {i+1} DX1 Contour')
        #     # Plot the contour of DX2 in the third column
        #     axs[1, i].contourf(X1, X2, DX2[i], cmap='viridis')
        #     axs[1, i].set_title(f'Batch {i+1} DX2 Contour')
        #     # Plot the vector field in the first column
        #     axs[2, i].quiver(X1, X2, DX1[i], DX2[i], angles='xy', scale_units='xy', scale=10, headlength=4, headaxislength=4, headwidth=4)
        #           # Plot the trajectory on top of the vector field
        #     axs[2, i].plot(traj_i[:, 0], traj_i[:, 1], 'r-')  # Plot the trajectory as a red line
        #     axs[2, i].plot(traj_i[0, 0], traj_i[0, 1], 'go')  # Mark the start of the trajectory with a green dot
        #     axs[2, i].plot(traj_i[-1, 0], traj_i[-1, 1], 'bo')  # Mark the end of the trajectory with a blue dot
        #     axs[2, i].set_xlim([-1.0, 1.0])
        #     axs[2, i].set_ylim([-1.0, 1.0])
        #     # Plot the trajectory in the second column
        #     axs[3, i].plot(t_samples_i, traj_i)
        #           # Set the titles of the subplots
        #     axs[2, i].set_title(f'Batch {i+1} Vector Field')
        #     axs[3, i].set_title(f'Batch {i+1} Trajectory')

        # self.fig = fig
        # self.axs = axs
              # Display the figure
        # plt.tight_layout()
        # plt.savefig('GPjax_ODE.png', dpi=300)
        # Loop over the subplots
        # for i in range(num_subplots):
        #     # Get the trajectory and time samples for this batch
        #     traj_i = y_samples[i, 0, :]  # Assuming the 1st trajectory is what you want
        #     t_samples_i = t_samples[i, 0, :, 0]  # Assuming the 1st initial condition
        #     # Plot the contour of DX1 in the second column
        #     axs[i, 0].contourf(X1, X2, DX1[i], cmap='viridis')
        #     axs[i, 0].set_title(f'Batch {i+1} DX1 Contour')
        #     # Plot the contour of DX2 in the third column
        #     axs[i, 1].contourf(X1, X2, DX2[i], cmap='viridis')
        #     axs[i, 1].set_title(f'Batch {i+1} DX2 Contour')
        #     # Plot the vector field in the first column
        #     axs[i, 2].quiver(X1, X2, DX1[i], DX2[i], angles='xy', scale_units='xy', scale=10, headlength=4, headaxislength=4, headwidth=4)
        #           # Plot the trajectory on top of the vector field
        #     axs[i, 2].plot(traj_i[:, 0], traj_i[:, 1], 'r-')  # Plot the trajectory as a red line
        #     axs[i, 2].plot(traj_i[0, 0], traj_i[0, 1], 'go')  # Mark the start of the trajectory with a green dot
        #     axs[i, 2].plot(traj_i[-1, 0], traj_i[-1, 1], 'bo')  # Mark the end of the trajectory with a blue dot
        #     axs[i, 2].set_xlim([-1.0, 1.0])
        #     axs[i, 2].set_ylim([-1.0, 1.0])
        #     # Plot the trajectory in the second column
        #     axs[i, 3].plot(t_samples_i, traj_i)
        #           # Set the titles of the subplots
        #     axs[i, 2].set_title(f'Batch {i+1} Vector Field')
        #     axs[i, 3].set_title(f'Batch {i+1} Trajectory')
        #       # Display the figure
        # plt.tight_layout()
        # plt.savefig('GPjax_ODE.png', dpi=300)

        # from matplotlib import pyplot as plt

        # plt.figure()
        # _, axis = plt.subplots(2, 1, figsize=(4, 6))
        # axis[0].plot(t_samples.T, np.squeeze(y_samples, -1).T, linewidth=0.5)
        # axis[0].set_title("ODE Samples from $df/dt=f(x)$")
        # context_data_point_x = np.array([0.8, 3.7])
        # context_data_point_y = np.array([0.4, 0.9])
        # samples_interp = vmap(np.interp, (None, 0, 0))(
        #     context_data_point_x, t_samples, np.squeeze(y_samples, -1)
        # )
        # axis[0].set_ylabel("State")
        # axis[0].set_xlabel("Time")
        # tol = 0.05
        # good_idx_mask = np.sum(np.abs(samples_interp - context_data_point_y), -1) < tol
        # axis[1].scatter(
        #     context_data_point_x, context_data_point_y, color="r", s=20, zorder=20
        # )
        # axis[1].plot(
        #     t_samples[good_idx_mask].T,
        #     np.squeeze(y_samples, -1)[good_idx_mask].T,
        #     linewidth=0.5,
        # )
        # axis[1].set_ylim(axis[0].get_ylim())
        # axis[1].set_title("ODE sample through context data")
        # axis[1].set_ylabel("State")
        # axis[1].set_xlabel("Time")
        # plt.tight_layout()
        # plt.savefig(f"1d_gp_ode_lengthscale{lengthscales}.png", dpi=300)
        # rng, step_rng = random.split(rng)
        # context_mask, target_mask = context_target_mask_gen(
        #     step_rng,
        #     batch_size=t_samples.shape[0],
        #     num_timesteps=num_all_data_points,
        #     num_context=num_context,
        #     num_extra_target=num_extra_target,
        # )
        # note np.zeros_like(y_samples) is only a placeholder for params
        return t_samples, y_samples, rng, np.zeros_like(y_samples) # context_mask, target_mask, 

    @staticmethod
    def sample_gp_ode_prior_for_plot(
        rng: random.PRNGKey,
        dynamic_sample_num: int,
        initial_cond_sample_num: int,
        num_timesteps: int,
        initial_condition_range: List,
        t_range: List,
        x_dim,
        lengthscale_range,
        signal_std_range,
        likelihood_std,
        time_scaling_coefficient: float = 1.0,
        num_context_range: Optional[List] = None,
        num_target_range: Optional[List] = None,
        use_initial: bool = False,
        specified_times: Optional[ArrayLike] = None,
        solver = 'Dopri5',
    ):

        rng, lengthscale_rng = random.split(rng)
        lengthscales = uniform(  # only support 1d right now
            lengthscale_rng,
            shape=(x_dim,),
            minval=np.asarray(lengthscale_range[0]),
            maxval=np.asarray(lengthscale_range[1]),
        )
        # assert len(signal_std_range) == 2
        rng, signal_std_rng = random.split(rng)
        signal_std = uniform(  # only support 1d right now
            signal_std_rng,
            shape=(1,),
            minval=signal_std_range[0],
            maxval=signal_std_range[1],
        )[0]
        signal_variance = signal_std**2

        sample_fns = [] 
        for _ in range(x_dim):
            kernel = jk.RBF(lengthscale=lengthscales, variance=signal_variance, active_dims=list(range(x_dim)))
            meanf = Zero()
            prior = Prior(mean_function=meanf, kernel=kernel)
            rng, gp_sample_rng = random.split(rng)
            sample_fn = prior.sample_approx(dynamic_sample_num, key=gp_sample_rng, num_features=512)
            sample_fns.append(sample_fn)

        # ode_fn = lambda t, x, args: sample_fn(x)
        @jit
        def dynamics(t, x, args):
            """
            :params x: [num_input, x_dim]
            """
            sample_id = args
            dxs = []
            for i in range(x_dim):
                dx = sample_fns[i](x)[..., sample_id]
                dxs.append(dx)
            return np.stack(dxs, axis=-1)
            # samples = sample_fn(x)
            # # return  # [num_input, num_samples]
            # return samples[..., sample_id]

        # sample initial state
        rng, x0_rng = random.split(rng)
        sample_x0 = random.uniform(
            x0_rng, 
            minval=np.asarray(initial_condition_range[0]), 
            maxval=np.asarray(initial_condition_range[1]), 
            shape=(dynamic_sample_num, initial_cond_sample_num, x_dim)
        )  # note the last dim is used for rff sample, bit hacky

        if specified_times is None:
            t_samples = np.linspace(t_range[0], t_range[1], num_timesteps)
        else:
            t_samples = specified_times
        raw_t_samples = (
            t_samples * time_scaling_coefficient
        )  # note this is used in NODEP paper that it scaled the time by 10
        raw_t_range = [t_range[0], time_scaling_coefficient * (t_range[1] - t_range[0])]

        # solve ode
        batch_diff_solve = lambda x0, ts, smp_idx: diffeqsolve(
            ODETerm(dynamics),
            t0=raw_t_range[0],
            t1=raw_t_range[1],
            dt0=None,
            solver=getattr(diffrax, solver)(),
            stepsize_controller=PIDController(rtol=1.4e-8, atol=1.4e-8),
            max_steps=5000,
            throw=False,
            y0=x0,
            saveat=SaveAt(ts=raw_t_samples),
            args=smp_idx,
        )

        # batch_diff_solve(sample_x0[0], np.sort(t_samples[0], axis=0)[:, 0], 0)
        f_samples = vmap(batch_diff_solve, (0, None, 0))(
            sample_x0, t_samples, np.arange(dynamic_sample_num)
        ).ys

        f_samples = rearrange(f_samples, "dynamics timestep traj_size state_dim -> dynamics traj_size timestep state_dim")
        t_samples = np.repeat(
            np.repeat(t_samples[None, ...], repeats=initial_cond_sample_num, axis=0)[
                None, ...
            ],
            repeats=dynamic_sample_num,
            axis=0,
        )[
            ..., None
        ]  # [dynamic_sample_num, initial_cond_sample_num, timesteps, 1]
        # dynamics_illustration = vmap(debug_ode_fn, (0, 0, 0))(sample_x0, t_samples, np.arange(batch_size))
        # observation noise
        rng, x0_rng = random.split(rng)
        y_samples = f_samples + likelihood_std * normal(
            x0_rng, shape=f_samples.shape
        )
        from matplotlib import pyplot as plt
        # # # # Create a grid of points to evaluate the vector field at
        plt.figure()
        x1 = np.linspace(-1, 1, 50)
        x2 = np.linspace(-1, 1, 50)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.repeat(np.stack((X1.flatten(), X2.flatten()), axis=-1)[None, ...], dynamic_sample_num, axis=0)
        DX = vmap(dynamics, in_axes=(None, 0, 0))(None, X, np.arange(dynamic_sample_num))
        DX1, DX2 = DX[..., 0].reshape(dynamic_sample_num, 50, 50), DX[..., 1].reshape(dynamic_sample_num, 50, 50)
        # Get the number of batches
        num_batches = dynamic_sample_num
        # Get the number of subplots to plot
        num_subplots = 6 # min(num_batches, 20)
        # Create a figure with a 2-column grid of subplots
        
        # fig, axs = plt.subplots(num_subplots, 4, figsize=(12, 4*num_subplots))
        fig, axs =  plt.subplots(6, 6, figsize=(12, 14)) # plt.subplots(num_subplots, 4, figsize=(2, 4*num_subplots))
        for i in range(num_subplots):
            # Get the trajectory and time samples for this batch
            traj_i = y_samples[i, 0, :]  # Assuming the 1st trajectory is what you want
            t_samples_i = t_samples[i, 0, :, 0]  # Assuming the 1st initial condition
            # Plot the contour of DX1 in the second column
            axs[0, i].contourf(X1, X2, DX1[i], cmap='viridis')
            axs[0, i].set_title(r'Sample $\mathbf{\mathit{f}}(\mathit{x})_1$', fontsize=8)
            axs[0, i].set_xticks([])
            axs[0, i].set_yticks([])
            # Plot the contour of DX2 in the third column
            axs[1, i].contourf(X1, X2, DX2[i], cmap='viridis')
            axs[1, i].set_title(r'Sample $\mathbf{\mathit{f}}(\mathit{x})_2$', fontsize=8)
            axs[1, i].set_xticks([])
            axs[1, i].set_yticks([])
            # Plot the vector field in the first column
            # axs[2, i].quiver(X1, X2, DX1[i], DX2[i], angles='xy', scale_units='xy', scale=20, headlength=4, headaxislength=4, headwidth=4)
            step = 3
            axs[2, i].quiver(
                    X1[::step, ::step], 
                    X2[::step, ::step], 
                    DX1[i][::step, ::step], 
                    DX2[i][::step, ::step],
                    angles='xy', 
                    scale_units='xy', 
                    scale=20, 
                    headlength=4, 
                    headaxislength=4, 
                    headwidth=4
                )
                                  # Plot the trajectory on top of the vector field
            axs[2, i].plot(traj_i[:, 0], traj_i[:, 1], 'r-')  # Plot the trajectory as a red line
            axs[2, i].plot(traj_i[0, 0], traj_i[0, 1], 'go')  # Mark the start of the trajectory with a green dot
            axs[2, i].plot(traj_i[-1, 0], traj_i[-1, 1], 'bo')  # Mark the end of the trajectory with a blue dot
            axs[2, i].set_xlim([-1.0, 1.0])
            axs[2, i].set_ylim([-1.0, 1.0])
            axs[2, i].set_xticks([])
            axs[2, i].set_yticks([])
            # Plot the trajectory in the second column
            axs[3, i].plot(t_samples_i, traj_i)
            axs[3, i].plot(t_samples_i[0], traj_i[0, 0], 'go')
            axs[3, i].plot(t_samples_i[0], traj_i[0, 1], 'go')
            axs[3, i].plot(t_samples_i[-1], traj_i[-1, 0], 'bo')
            axs[3, i].plot(t_samples_i[-1], traj_i[-1, 1], 'bo')
                  # Set the titles of the subplots
            axs[2, i].set_title(f'Vector Field', fontsize=8)
            axs[3, i].set_title(f'Sample Trajectory', fontsize=8)
            axs[3, i].set_xlabel('Time', fontsize=8)
            axs[3, i].set_ylabel('State Value', fontsize=8)
            axs[3, i].set_xticks([])
            axs[3, i].set_yticks([])

        return t_samples, y_samples, rng, fig, axs

    def get_data_input_scaler(self, inputs):
        return inputs

    def get_data_inverse_scaler(self, inputs):
        return inputs

    def get_aux_datasets(self):
        return self.aux_datasets


class LOTKA_VOLTERRA_ODE(TFDataset):
    """ "
    Implementation of the LOTKA VOLTERRA System
    refer https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations

    a reference implementation is also provided in Nueral ODE Process repo at
    https://github.com/crisbodnar/ndp/blob/main/data/datasets.py

    some information are also provided in openreview
    https://openreview.net/forum?id=27acGyyI1BY

    Parameters
    ----------
    alpha_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.

    beta_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled.

    delta_range : int
        Number of samples of the function contained in dataset.

    gamma_range : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].

    t_range : List of float
        Defines the range from which the time is sampled.


    """

    def generate_tf_dataset(
        self,
        data_gen_rng: random.PRNGKey,
        x_0_range: List,
        alpha_range: List,
        beta_range: List,
        delta_range: List,
        gamma_range: List,
        t_range: Optional[List] = [0, 1.5],
        num_context_range: Optional[List] = None,
        num_timesteps: Optional[int] = None,
        num_target_range: Optional[int] = None,
        aux: Optional[dict] = None,
        generator: bool = True,
        dynamics_smp_num: int = 20,
        initial_condition_smp_num: int = 20,
        num_train_samples: Optional[int] = np.inf,
        batch_size: Optional[int] = np.inf,
        generator_use_initial: bool = True,
        time_scaling_coefficient: float = 10.0,
    ) -> tf.data.Dataset:
        """
        :params data_gen_rng: the random number generator
        :params x_0_range: the range of the initial condition
        :params alpha_range: the range of the alpha parameter
        :params beta_range: the range of the beta parameter
        :param delta_range: the number of delta parameter
        :param gamma_range: the number of gamma parameter
        :param t_range: the range of the time, NOTE that within the ODE, it has been hard coded to scale by 10 as done in Neural ODE Process paper
        :param num_context_range: the number of context points to be sampled
        :param num_timesteps: the number of total timesteps to be sampled, this will be uniformely spaced
        :param num_target_range: the number of target points to be sampled
        :param aux: dict the auxiliary datasets to be generated: {dataset_name: (num_aug_dynamic, num_aug_initial_cond)}
        :param generator: bool, whether to return a generator or a tf.data.Dataset
        :param dynamics_smp_num the number of dynamic systems to sample
        :param initial_condition_smp_num the number of initial conditions to sample within each dynamic systems
        :param num_train_samples: the number of training samples to be generated
        :param batch_size TODO
        :param generator_use_initial: bool, whether to use the initial condition as the context points
        """
        if generator is True:
            self._is_generator = True
        # note that this will make all the generated aux datasets the same since we do not change the rng,
        # this is ideal if we want to compare between different models as no need to store same data locally
        if aux is not None:
            data_gen_rng, aux_data_gen_rng = random.split(data_gen_rng, 2)
            aux_datasets = {}
            for key, (num_aug_dynamic, num_aug_initial_cond) in aux.items():
                # , alpha_smps, beta_smps, delta_smps, gamma_smps
                x_samples, y_samples, _, params = self.sample_lotka_volterra_trajectory(
                    aux_data_gen_rng,
                    dynamic_sample_num=num_aug_dynamic,
                    initial_cond_sample_num=num_aug_initial_cond,
                    num_timesteps=num_timesteps,
                    initial_condition_range=tuple(
                        tuple(_x_0_range) for _x_0_range in x_0_range
                    ),
                    t_range=t_range,
                    alpha_range=alpha_range,
                    beta_range=beta_range,
                    delta_range=delta_range,
                    gamma_range=gamma_range,
                    use_initial=generator_use_initial,
                    num_context_range=num_context_range,
                    num_target_range=num_target_range,
                    time_scaling_coefficient = time_scaling_coefficient,
                )
                # if generator == True:
                #     aux_datasets[key] = (x_samples, y_samples, context_mask, target_mask)
                # else:
                # aux_datasets[key] = (x_samples, y_samples, (alpha_smps, beta_smps, delta_smps, gamma_smps))
                aux_datasets[key] = (x_samples, y_samples, params)
        else:
            aux_datasets = {}
        self.aux_datasets = aux_datasets

        # from matplotlib import pyplot as plt
        # plt.figure()
        # _, axs = plt.subplots(nrows=5, ncols=1, figsize=(3, 10))
        # for j in range(5):
        #     for i in range(num_aug_dynamic):
        #         axs[j].plot(x_samples[j, i], y_samples[j, i][..., 0], color='r')
        #         # axs[j].scatter(x_samples[j, i], y_samples[j, i][..., 0], s=10, color='r')
        #         axs[j].plot(x_samples[j, i], y_samples[j, i][..., 1], color='b')
        #         # axs[j].scatter(x_samples[j, i], y_samples[j, i][..., 1], s=10, color='b')
        # plt.xlabel('time')
        # plt.ylabel('population')
        # plt.suptitle('lotka volterra sample trajectories')
        # plt.savefig('lotka_volterra.png', dpi=300)

        def batch_training_data_generator(rng, batch_size):
            while True: # , alpha_smps, beta_smps, delta_smps, gamma_smps
                x_samples, y_samples, rng, params = self.sample_lotka_volterra_trajectory(
                    rng,
                    dynamic_sample_num=dynamics_smp_num,
                    initial_cond_sample_num=initial_condition_smp_num,
                    num_timesteps=num_timesteps,
                    initial_condition_range=x_0_range,
                    t_range=t_range,
                    alpha_range=alpha_range,
                    beta_range=beta_range,
                    delta_range=delta_range,
                    gamma_range=gamma_range,
                    use_initial=generator_use_initial,
                    num_context_range=num_context_range,
                    num_target_range=num_target_range,
                    time_scaling_coefficient=time_scaling_coefficient,
                )
                yield x_samples, y_samples, params # , (alpha_smps, beta_smps, delta_smps, gamma_smps)  # , context_mask, target_mask # [..., N, 1]

        if generator:
            data_gen = lambda: batch_training_data_generator(data_gen_rng, batch_size)
            train_data = tf.data.Dataset.from_generator(
                data_gen, (np.float32, np.float32, np.float32)
            )  # # , np.bool_, np.bool_))
            return train_data
        else: # , alpha_smps, beta_smps, delta_smps, gamma_smps
            x_samples, y_samples, rng, params = self.sample_lotka_volterra_trajectory(
                data_gen_rng,
                dynamic_sample_num=dynamics_smp_num,
                initial_cond_sample_num=initial_condition_smp_num,
                num_timesteps=num_timesteps,
                initial_condition_range=x_0_range,
                t_range=t_range,
                alpha_range=alpha_range,
                beta_range=beta_range,
                delta_range=delta_range,
                gamma_range=gamma_range,
                use_initial=generator_use_initial,
                num_context_range=num_context_range,
                num_target_range=num_target_range,
                time_scaling_coefficient = time_scaling_coefficient
            )
            train_data = tf.data.Dataset.from_tensor_slices((x_samples, y_samples, params))
            # train_data = tf.data.Dataset.from_tensor_slices((x_samples, y_samples, (alpha_smps, beta_smps, delta_smps, gamma_smps)))
            return train_data

    @staticmethod
    @partial(
        jit,
        static_argnames=(
            "dynamic_sample_num",
            "initial_cond_sample_num",
            "num_timesteps",
            "initial_condition_range",
            "t_range",
            "alpha_range",
            "beta_range",
            "delta_range",
            "gamma_range",
            "num_context_range",
            "num_target_range",
            "use_initial",
            "specified_times",
            "time_scaling_coefficient"
        ),
    )
    def sample_lotka_volterra_trajectory(
        rng: random.PRNGKey,
        dynamic_sample_num: int,
        initial_cond_sample_num: int,
        num_timesteps: int,
        initial_condition_range: List,
        t_range: List,
        alpha_range: List,
        beta_range: List,
        delta_range: List,
        gamma_range: List,
        num_context_range: Optional[List] = None,
        num_target_range: Optional[List] = None,
        use_initial: bool = False,
        specified_times: Optional[ArrayLike] = None,
        time_scaling_coefficient: float = 10.0,
    ):
        """
        generate the data from the lotka volterra system

        this function supports two approaches, one is used as a fixed dataset, the other one is use as a generator

        :param rng: the random number generator
        """
        rng, initial_rng, alpha_rng, beta_rng, delta_rng, gamma_rng = random.split(
            rng, 6
        )
        # 2024/2/18: we have changed here to be make it more general, e.g., we can start with 2 initial condition
        x0 = random.uniform(
            initial_rng,
            shape=(dynamic_sample_num, initial_cond_sample_num, 2),
            minval=np.asarray(initial_condition_range[0]),
            maxval=np.asarray(initial_condition_range[1]),
        )  # [dynamic_smp, ibatch_size, 1]
        # E = random.uniform(initial_rng, shape=(dynamic_sample_num, initial_cond_sample_num, 1), \
        #                    minval=initial_condition_range[0], \
        #                    maxval=initial_condition_range[1]) # [dynamic_smp, ibatch_size, 1]
        # x0 = np.concatenate([2 * E, E], axis=-1) # [dynamic_smp, ibatch_size, 2]

        alpha = np.repeat(
            random.uniform(
                alpha_rng,
                shape=(dynamic_sample_num,),
                minval=alpha_range[0],
                maxval=alpha_range[1],
            )[..., None],
            initial_cond_sample_num,
            axis=-1,
        )  # [dynamic_smp]
        beta = np.repeat(
            random.uniform(
                beta_rng,
                shape=(dynamic_sample_num,),
                minval=beta_range[0],
                maxval=beta_range[1],
            )[..., None],
            initial_cond_sample_num,
            axis=-1,
        )  # [dynamic_smp]
        delta = np.repeat(
            random.uniform(
                delta_rng,
                shape=(dynamic_sample_num,),
                minval=delta_range[0],
                maxval=delta_range[1],
            )[..., None],
            initial_cond_sample_num,
            axis=-1,
        )  # [dynamic_smp]
        gamma = np.repeat(
            random.uniform(
                gamma_rng,
                shape=(dynamic_sample_num,),
                minval=gamma_range[0],
                maxval=gamma_range[1],
            )[..., None],
            initial_cond_sample_num,
            axis=-1,
        )  # [dynamic_smp]

        def dynamics(t, _x, args):
            u, v = _x[..., 0], _x[..., 1]
            return np.stack(
                [alpha * u - beta * u * v, delta * u * v - gamma * v], axis=-1
            )

        if specified_times is None:
            t_samples = np.linspace(t_range[0], t_range[1], num_timesteps)
        else:
            t_samples = specified_times
        raw_t_samples = (
            t_samples * time_scaling_coefficient
        )  # note this is used in NODEP paper that it scaled the time by 10
        raw_t_range = [t_range[0], time_scaling_coefficient * (t_range[1] - t_range[0])]
        rtol, atol = 1e-7, 1e-9
        while True:
            try:
                # solve ode
                traj = diffeqsolve(
                    ODETerm(dynamics),
                    t0=raw_t_range[0],
                    t1=raw_t_range[1],
                    dt0=None,
                    stepsize_controller=PIDController(rtol=rtol, atol=atol),
                    y0=x0,
                    solver=Dopri5(),
                    max_steps=5000,
                    throw=False,
                    saveat=SaveAt(ts=raw_t_samples),
                ).ys
                break
            except Exception as e:
                print('Data generator encountered ODE solving issue, using a larger rtol and atl')
                rtol *= 10
                atol *= 10
        traj = rearrange(traj, "times batch traj_size dim -> batch traj_size times dim")
        t_samples = np.repeat(
            np.repeat(t_samples[None, ...], repeats=initial_cond_sample_num, axis=0)[
                None, ...
            ],
            repeats=dynamic_sample_num,
            axis=0,
        )[
            ..., None
        ]  # [dynamic_sample_num, initial_cond_sample_num, timesteps, 1]
        context_mask = target_mask = None

        # from matplotlib import pyplot as plt
        # fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
        #
        # for i in range(25):  # We will create 25 plots
        #     combined = np.concatenate([np.repeat(E[i], traj[i].shape[-2], axis=-1)[..., None], traj[i, ..., :1]], axis=-1)
        #     # Sort combined by the ratio
        #     combined_sorted = combined[combined[...,0, 0].argsort()]
        #     # Separate traj from the sorted combined dataset
        #     traj_sorted = combined_sorted[..., 1]
        #
        #     # Create a meshgrid
        #     X, Y = np.meshgrid(t_samples[i][0, :, 0], combined_sorted[:, 0, 0])
        #
        #     # Adjust the position of each subplot to prevent the y-axes from overlapping
        #     box = axes[i//5, i%5].get_position()
        #     axes[i//5, i%5].set_position([box.x0, box.y0, box.width * 0.9, box.height])  # Adjust the position and width of each subplot
        #
        #     contour1 = axes[i//5, i%5].contourf(X, Y, traj_sorted)
        #     # for _init, traj in zip(init, traj_dicts.values()):
        #     #     axes[i//5, i%5].scatter(traj.times, np.repeat(_init, len(traj.times)), s=20, color='r', label='Real State at different trajectory')
        #
        #     axes[i//5, i%5].set_title('State 1 Value')
        #     axes[i//5, i%5].set_xlabel('Time')
        #     axes[i//5, i%5].set_ylabel('Initial Condition')
        # plt.suptitle('Lotka Volterra State 1 Value vs Time & Initial Conidtion')
        # plt.tight_layout()
        # # Save the figure with a unique name
        # plt.savefig('Lotka_Volterra_MultiPanel.png', dpi=300)
        #
        #
        return t_samples, traj, rng, np.concatenate([alpha[..., None], beta[..., None], delta[..., None], gamma[..., None]], axis=-1)


    def get_data_input_scaler(self, inputs):
        return inputs

    def get_data_inverse_scaler(self, inputs):
        return inputs

    def get_aux_datasets(self):
        return self.aux_datasets

    @property
    def is_generator(self):
        return self._is_generator


class LOTKA_VOLTERRA_ODE_3D(TFDataset):
    """
    Implementation of the 3D LOTKA VOLTERRA System with interactions among three species.
    This model extends the classic predator-prey model to include a third species.
    """

    def generate_tf_dataset(
        self,
        data_gen_rng: random.PRNGKey,
        x_0_range: List,
        alpha_range: List,
        beta_range: List,
        gamma_range: List,
        delta_range: List,
        epsilon_range: List,
        zeta_range: List,
        eta_range: List,
        theta_range: List,
        t_range: Optional[List] = [0, 1.5],
        num_context_range: Optional[List] = None,
        num_timesteps: Optional[int] = None,
        num_target_range: Optional[int] = None,
        aux: Optional[dict] = None,
        generator: bool = True,
        dynamics_smp_num: int = 20,
        initial_condition_smp_num: int = 20,
        num_train_samples: Optional[int] = np.inf,
        batch_size: Optional[int] = np.inf,
        generator_use_initial: bool = True,
        time_scaling_coefficient: float = 10.0,
    ) -> tf.data.Dataset:
        if generator is True:
            self._is_generator = True
        # note that this will make all the generated aux datasets the same since we do not change the rng,
        # this is ideal if we want to compare between different models as no need to store same data locally
        if aux is not None:
            data_gen_rng, aux_data_gen_rng = random.split(data_gen_rng, 2)
            aux_datasets = {}
            for key, (num_aug_dynamic, num_aug_initial_cond) in aux.items():
                # , alpha_smps, beta_smps, delta_smps, gamma_smps
                x_samples, y_samples, _, params = self.sample_lotka_volterra_trajectory_3D(
                    aux_data_gen_rng,
                    dynamic_sample_num=num_aug_dynamic,
                    initial_cond_sample_num=num_aug_initial_cond,
                    num_timesteps=num_timesteps,
                    initial_condition_range=tuple(
                        tuple(_x_0_range) for _x_0_range in x_0_range
                    ),
                    t_range=t_range,
                    alpha_range=alpha_range,
                    beta_range=beta_range,
                    delta_range=delta_range,
                    gamma_range=gamma_range,
                    use_initial=generator_use_initial,
                    num_context_range=num_context_range,
                    num_target_range=num_target_range,
                    time_scaling_coefficient = time_scaling_coefficient,
                    epsilon_range=epsilon_range,
                    zeta_range=zeta_range,
                    eta_range=eta_range,
                    theta_range=theta_range,
                )
                aux_datasets[key] = (x_samples, y_samples, params)
        else:
            aux_datasets = {}
        self.aux_datasets = aux_datasets

        # from matplotlib import pyplot as plt
        # plt.figure()
        # _, axs = plt.subplots(nrows=5, ncols=1, figsize=(3, 10))
        # for j in range(5):
        #     for i in range(num_aug_dynamic):
        #         axs[j].plot(x_samples[j, i], y_samples[j, i][..., 0], color='r')
        #         # axs[j].scatter(x_samples[j, i], y_samples[j, i][..., 0], s=10, color='r')
        #         axs[j].plot(x_samples[j, i], y_samples[j, i][..., 1], color='b')
        #         # axs[j].scatter(x_samples[j, i], y_samples[j, i][..., 1], s=10, color='b')
        # plt.xlabel('time')
        # plt.ylabel('population')
        # plt.suptitle('lotka volterra sample trajectories')
        # plt.savefig('lotka_volterra.png', dpi=300)

        def batch_training_data_generator(rng, batch_size):
            while True: # , alpha_smps, beta_smps, delta_smps, gamma_smps
                x_samples, y_samples, rng, params = self.sample_lotka_volterra_trajectory_3D(
                    rng,
                    dynamic_sample_num=dynamics_smp_num,
                    initial_cond_sample_num=initial_condition_smp_num,
                    num_timesteps=num_timesteps,
                    initial_condition_range=x_0_range,
                    t_range=t_range,
                    alpha_range=alpha_range,
                    beta_range=beta_range,
                    delta_range=delta_range,
                    gamma_range=gamma_range,
                    epsilon_range=epsilon_range,
                    zeta_range=zeta_range,
                    eta_range=eta_range,
                    theta_range=theta_range,
                    use_initial=generator_use_initial,
                    num_context_range=num_context_range,
                    num_target_range=num_target_range,
                    time_scaling_coefficient=time_scaling_coefficient
                )
                yield x_samples, y_samples, params

        if generator:
            data_gen = lambda: batch_training_data_generator(data_gen_rng, batch_size)
            train_data = tf.data.Dataset.from_generator(
                    data_gen, (np.float32, np.float32, np.float32)
                )
            return train_data
        else: # , alpha_smps, beta_smps, delta_smps, gamma_smps
            x_samples, y_samples, rng, params = self.sample_lotka_volterra_trajectory_3D(
                data_gen_rng,
                dynamic_sample_num=dynamics_smp_num,
                initial_cond_sample_num=initial_condition_smp_num,
                num_timesteps=num_timesteps,
                initial_condition_range=x_0_range,
                t_range=t_range,
                alpha_range=alpha_range,
                beta_range=beta_range,
                delta_range=delta_range,
                gamma_range=gamma_range,
                epsilon_range=epsilon_range,
                zeta_range=zeta_range,
                eta_range=eta_range,
                theta_range=theta_range,
                use_initial=generator_use_initial,
                num_context_range=num_context_range,
                num_target_range=num_target_range,
                time_scaling_coefficient = time_scaling_coefficient
            )
            train_data = tf.data.Dataset.from_tensor_slices((x_samples, y_samples, params))
            # train_data = tf.data.Dataset.from_tensor_slices((x_samples, y_samples, (alpha_smps, beta_smps, delta_smps, gamma_smps)))
            return train_data


    @staticmethod
    @partial(
        jit,
        static_argnames=(
            "dynamic_sample_num",
            "initial_cond_sample_num",
            "num_timesteps",
            "initial_condition_range",
            "t_range",
            "alpha_range",
            "beta_range",
            "delta_range",
            "gamma_range",
            "epsilon_range",
            "zeta_range",
            "eta_range",
            "theta_range",
            "num_context_range",
            "num_target_range",
            "use_initial",
            "specified_times",
        ),
    )
    def sample_lotka_volterra_trajectory_3D(
        rng: random.PRNGKey,
        dynamic_sample_num: int,
        initial_cond_sample_num: int,
        num_timesteps: int,
        initial_condition_range: List,
        t_range: List,
        alpha_range: List,
        beta_range: List,
        gamma_range: List,
        delta_range: List,
        epsilon_range: List,
        zeta_range: List,
        eta_range: List,
        theta_range: List,
        num_context_range: Optional[List] = None,
        num_target_range: Optional[List] = None,
        use_initial: bool = False,
        specified_times: Optional[ArrayLike] = None,
        time_scaling_coefficient: float = 10.0,
    ):
        rng, initial_rng, alpha_rng, beta_rng, delta_rng, gamma_rng, epsilon_rng, zeta_rng, eta_rng, theta_rng = random.split(
            rng, 10
        )
        # 2024/2/18: we have changed here to be make it more general, e.g., we can start with 2 initial condition
        x0 = random.uniform(
            initial_rng,
            shape=(dynamic_sample_num, initial_cond_sample_num, 3),
            minval=np.asarray(initial_condition_range[0]),
            maxval=np.asarray(initial_condition_range[1]),
        )  # [dynamic_smp, ibatch_size, 1]
        alpha = np.repeat(
            random.uniform(
                alpha_rng,
                shape=(dynamic_sample_num,),
                minval=alpha_range[0],
                maxval=alpha_range[1],
            )[..., None],
            initial_cond_sample_num,
            axis=-1,
        )  # [dynamic_smp]
        beta = np.repeat(
            random.uniform(
                beta_rng,
                shape=(dynamic_sample_num,),
                minval=beta_range[0],
                maxval=beta_range[1],
            )[..., None],
            initial_cond_sample_num,
            axis=-1,
        )  # [dynamic_smp]
        delta = np.repeat(
            random.uniform(
                delta_rng,
                shape=(dynamic_sample_num,),
                minval=delta_range[0],
                maxval=delta_range[1],
            )[..., None],
            initial_cond_sample_num,
            axis=-1,
        )  # [dynamic_smp]
        epsilon = np.repeat(
            random.uniform(
                epsilon_rng,
                shape=(dynamic_sample_num,),
                minval=epsilon_range[0],
                maxval=epsilon_range[1],
            )[..., None],
            initial_cond_sample_num,
            axis=-1,
        )   
        zeta = np.repeat(
            random.uniform(
                zeta_rng,
                shape=(dynamic_sample_num,),
                minval=zeta_range[0],
                maxval=zeta_range[1],
            )[..., None],
            initial_cond_sample_num,
            axis=-1,
        )
        eta = np.repeat(
            random.uniform(
                eta_rng,
                shape=(dynamic_sample_num,),
                minval=eta_range[0],
                maxval=eta_range[1],
            )[..., None],
            initial_cond_sample_num,
            axis=-1,
        )
        theta = np.repeat(
            random.uniform(
                theta_rng,
                shape=(dynamic_sample_num,),
                minval=theta_range[0],
                maxval=theta_range[1],
            )[..., None],
            initial_cond_sample_num,
            axis=-1,
        )

        gamma = np.repeat(
            random.uniform(
                gamma_rng,
                shape=(dynamic_sample_num,),
                minval=gamma_range[0],
                maxval=gamma_range[1],
            )[..., None],
            initial_cond_sample_num,
            axis=-1,
        )  # [dynamic_smp]

        def dynamics(t, _x, args):
            x_1, x_2, x_3 = _x[..., 0], _x[..., 1], _x[..., 2]
            return np.stack([
                alpha * x_1 - beta * x_1 * x_2 - epsilon * x_1 * x_3,
                delta * x_1 * x_2 - gamma * x_2 - zeta * x_2 * x_3,
                eta * x_1 * x_3 - theta * x_3
                # eta * x_1 * x_3 + theta * x_2 * x_3 - gamma * x_3
            ], axis=-1)
            # return np.stack(
            #     [alpha * u - beta * u * v, delta * u * v - gamma * v], axis=-1
            # )

        if specified_times is None:
            t_samples = np.linspace(t_range[0], t_range[1], num_timesteps)
        else:
            t_samples = specified_times
        raw_t_samples = (
            t_samples * time_scaling_coefficient
        )  # note this is used in NODEP paper that it scaled the time by 10
        raw_t_range = [t_range[0], time_scaling_coefficient * (t_range[1] - t_range[0])]
        rtol, atol = 1e-7, 1e-9
        while True:
            try:
                # solve ode
                traj = diffeqsolve(
                    ODETerm(dynamics),
                    t0=raw_t_range[0],
                    t1=raw_t_range[1],
                    dt0=None,
                    stepsize_controller=PIDController(rtol=rtol, atol=atol),
                    y0=x0,
                    solver=Dopri5(),
                    max_steps=5000,
                    throw=False,
                    saveat=SaveAt(ts=raw_t_samples),
                ).ys
                break
            except Exception as e:
                print('Data generator encountered ODE solving issue, using a larger rtol and atl')
                rtol *= 10
                atol *= 10
        traj = rearrange(traj, "times batch traj_size dim -> batch traj_size times dim")
        t_samples = np.repeat(
            np.repeat(t_samples[None, ...], repeats=initial_cond_sample_num, axis=0)[
                None, ...
            ],
            repeats=dynamic_sample_num,
            axis=0,
        )[
            ..., None
        ]  # [dynamic_sample_num, initial_cond_sample_num, timesteps, 1]
        context_mask = target_mask = None

        return t_samples, traj, rng, np.stack([alpha, beta, epsilon, delta, gamma, zeta, eta, theta], axis=-1) # , alpha, beta, delta, gamma

    def get_data_input_scaler(self, inputs):
        return inputs

    def get_data_inverse_scaler(self, inputs):
        return inputs

    def get_aux_datasets(self):
        return self.aux_datasets

    @property
    def is_generator(self):
        return self._is_generator


class Brusselator(TFDataset):
    """
    Parameters
    ----------
    b_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.
    A_range : tuple of float
    """

    def generate_tf_dataset(
        self,
        data_gen_rng: random.PRNGKey,
        x_0_range: List,
        A_range: List,
        B_range: List,
        t_range: Optional[List] = [0, 1.5],
        num_context_range: Optional[List] = None,
        num_timesteps: Optional[int] = None,
        num_target_range: Optional[int] = None,
        aux: Optional[dict] = None,
        generator: bool = True,
        dynamics_smp_num: int = 20,
        initial_condition_smp_num: int = 20,
        num_train_samples: Optional[int] = np.inf,
        batch_size: Optional[int] = np.inf,
        generator_use_initial: bool = True,
        time_scaling_coefficient: float = 10.0,
    ) -> tf.data.Dataset:
        """
        :params data_gen_rng: the random number generator
        :params x_0_range: the range of the initial condition
        :param t_range: the range of the time, NOTE that within the ODE, it has been hard coded to scale by 10 as done in Neural ODE Process paper
        :param num_context_range: the number of context points to be sampled
        :param num_timesteps: the number of total timesteps to be sampled, this will be uniformely spaced
        :param num_target_range: the number of target points to be sampled
        :param aux: dict the auxiliary datasets to be generated: {dataset_name: (num_aug_dynamic, num_aug_initial_cond)}
        :param generator: bool, whether to return a generator or a tf.data.Dataset
        :param dynamics_smp_num the number of dynamic systems to sample
        :param initial_condition_smp_num the number of initial conditions to sample within each dynamic systems
        :param num_train_samples: the number of training samples to be generated
        :param batch_size TODO
        :param generator_use_initial: bool, whether to use the initial condition as the context points
        """
        if generator is True:
            self._is_generator = True
        # note that this will make all the generated aux datasets the same since we do not change the rng,
        # this is ideal if we want to compare between different models as no need to store same data locally
        if aux is not None:
            data_gen_rng, aux_data_gen_rng = random.split(data_gen_rng, 2)
            aux_datasets = {}
            for key, (num_aug_dynamic, num_aug_initial_cond) in aux.items():
                x_samples, y_samples, _, params = self.sample_brusselator_trajectory(
                    aux_data_gen_rng,
                    dynamic_sample_num=num_aug_dynamic,
                    initial_cond_sample_num=num_aug_initial_cond,
                    num_timesteps=num_timesteps,
                    initial_condition_range=tuple(
                        tuple(_x_0_range) for _x_0_range in x_0_range
                    ),
                    t_range=t_range,
                    A_range=A_range,
                    B_range = B_range,
                    use_initial=generator_use_initial,
                    num_context_range=num_context_range,
                    num_target_range=num_target_range,
                    time_scaling_coefficient = time_scaling_coefficient
                )
                # if generator == True:
                #     aux_datasets[key] = (x_samples, y_samples, context_mask, target_mask)
                # else:
                aux_datasets[key] = (x_samples, y_samples, params)
        else:
            aux_datasets = {}
        self.aux_datasets = aux_datasets

        # from matplotlib import pyplot as plt
        # plt.figure()
        # _, axs = plt.subplots(nrows=5, ncols=1, figsize=(3, 10))
        # for j in range(5):
        #     for i in range(num_aug_dynamic):
        #         axs[j].plot(x_samples[j, i], y_samples[j, i][..., 0], color='r')
        #         # axs[j].scatter(x_samples[j, i], y_samples[j, i][..., 0], s=10, color='r')
        #         axs[j].plot(x_samples[j, i], y_samples[j, i][..., 1], color='b')
        #         # axs[j].scatter(x_samples[j, i], y_samples[j, i][..., 1], s=10, color='b')
        # plt.xlabel('time')
        # plt.ylabel('population')
        # plt.suptitle('lotka volterra sample trajectories')
        # plt.savefig('lotka_volterra.png', dpi=300)

        def batch_training_data_generator(rng, batch_size):
            while True:
                x_samples, y_samples, rng, params = self.sample_brusselator_trajectory(
                    rng,
                    dynamic_sample_num=dynamics_smp_num,
                    initial_cond_sample_num=initial_condition_smp_num,
                    num_timesteps=num_timesteps,
                    initial_condition_range=x_0_range,
                    t_range=t_range,
                    A_range=A_range,
                    B_range = B_range,
                    use_initial=generator_use_initial,
                    num_context_range=num_context_range,
                    num_target_range=num_target_range,
                    time_scaling_coefficient=time_scaling_coefficient
                )
                yield x_samples, y_samples, params  # , context_mask, target_mask # [..., N, 1]

        if generator:
            data_gen = lambda: batch_training_data_generator(data_gen_rng, batch_size)
            train_data = tf.data.Dataset.from_generator(
                    data_gen, (np.float32, np.float32, np.float32)
                )
            return train_data
        else:
            x_samples, y_samples, rng, params = self.sample_brusselator_trajectory(
                data_gen_rng,
                dynamic_sample_num=dynamics_smp_num,
                initial_cond_sample_num=initial_condition_smp_num,
                num_timesteps=num_timesteps,
                initial_condition_range=x_0_range,
                t_range=t_range,
                A_range=A_range,
                B_range = B_range,
                use_initial=generator_use_initial,
                num_context_range=num_context_range,
                num_target_range=num_target_range,
                time_scaling_coefficient=time_scaling_coefficient
            )
            train_data = tf.data.Dataset.from_tensor_slices((x_samples, y_samples, params))
            return train_data

    @staticmethod
    @partial(
        jit,
        static_argnames=(
            "dynamic_sample_num",
            "initial_cond_sample_num",
            "num_timesteps",
            "initial_condition_range",
            "t_range",
            "A_range",
            "B_range",
            "num_context_range",
            "num_target_range",
            "use_initial",
            "specified_times",
            "time_scaling_coefficient", 
        ),
    )
    def sample_brusselator_trajectory(
        rng: random.PRNGKey,
        dynamic_sample_num: int,
        initial_cond_sample_num: int,
        num_timesteps: int,
        initial_condition_range: List,
        t_range: List,
        A_range: List,
        B_range: List,
        num_context_range: Optional[List] = None,
        num_target_range: Optional[List] = None,
        use_initial: bool = False,
        specified_times: Optional[ArrayLike] = None,
        time_scaling_coefficient: float = 10.0,
    ):
        """
        generate the data from the lotka volterra system

        this function supports two approaches, one is used as a fixed dataset, the other one is use as a generator

        :param rng: the random number generator
        """
        rng, initial_rng, r_rng, A_rng = random.split(
            rng, 4
        )
        num_states = len(initial_condition_range[0])
        x0 = random.uniform(    
            initial_rng,
            shape=(dynamic_sample_num, initial_cond_sample_num, num_states),
            minval=np.asarray(initial_condition_range[0]),
            maxval=np.asarray(initial_condition_range[1]),
        )  # [dynamic_smp, ibatch_size, 1]

        B = random.uniform(r_rng, shape=(dynamic_sample_num, 1), minval=B_range[0], maxval=B_range[1])
        A = random.uniform(A_rng, shape=(dynamic_sample_num, 1), minval=A_range[0], maxval=A_range[1])
        A = np.repeat(np.expand_dims(A, axis=-2), initial_cond_sample_num, axis=-2)
        B = np.repeat(np.expand_dims(B, axis=-2), initial_cond_sample_num, axis=-2)
        
        def dynamics(t, x, args):
            _x, _y = np.split(x, 2, axis=-1)
            _dx = A + (_x ** 2) * _y - _x * (B + 1)
            _dy = B * _x - (_x ** 2) * _y 
            return np.concatenate([_dx, _dy], axis=-1)

        if specified_times is None:
            t_samples = np.linspace(t_range[0], t_range[1], num_timesteps)
        else:
            t_samples = specified_times
        raw_t_samples = (
            t_samples * time_scaling_coefficient
        )  # note this is used in NODEP paper that it scaled the time by 10
        raw_t_range = [t_range[0], time_scaling_coefficient * (t_range[1] - t_range[0])]

        # raw_t_samples = (
        #     t_samples * 10
        # )  # note this is used in NODEP paper that it scaled the time by 10
        # raw_t_range = [t_range[0], 10 * (t_range[1] - t_range[0])]
        rtol, atol = 1e-7, 1e-9
        while True:
            try:
                # solve ode
                traj = diffeqsolve(
                    ODETerm(dynamics),
                    t0=raw_t_range[0],
                    t1=raw_t_range[1],
                    dt0=None,
                    stepsize_controller=PIDController(rtol=rtol, atol=atol),
                    y0=x0,
                    solver=Dopri5(),
                    max_steps=5000,
                    throw=False,
                    saveat=SaveAt(ts=raw_t_samples),
                ).ys
                break
            except Exception as e:
                print('Data generator encountered ODE solving issue, using a larger rtol and atl')
                rtol *= 10
                atol *= 10
        traj = rearrange(traj, "times batch traj_size dim -> batch traj_size times dim")

        t_samples = np.repeat(
            np.repeat(t_samples[None, ...], repeats=initial_cond_sample_num, axis=0)[
                None, ...
            ],
            repeats=dynamic_sample_num,
            axis=0,
        )[
            ..., None
        ]  # [dynamic_sample_num, initial_cond_sample_num, timesteps, 1]
        context_mask = target_mask = None
        return t_samples, traj, rng, np.concatenate([A, B], axis=-1)

    def get_data_input_scaler(self, inputs):
        return inputs

    def get_data_inverse_scaler(self, inputs):
        return inputs

    def get_aux_datasets(self):
        return self.aux_datasets

    @property
    def is_generator(self):
        return self._is_generator


class SELKOV(TFDataset):
    """
    SELKOV model
    """
    """
    Parameters
    ----------
    b_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.
    a_range : tuple of float
    """

    def generate_tf_dataset(
        self,
        data_gen_rng: random.PRNGKey,
        x_0_range: List,
        a_range: List,
        b_range: List,
        t_range: Optional[List] = [0, 1.5],
        num_context_range: Optional[List] = None,
        num_timesteps: Optional[int] = None,
        num_target_range: Optional[int] = None,
        aux: Optional[dict] = None,
        generator: bool = True,
        dynamics_smp_num: int = 20,
        initial_condition_smp_num: int = 20,
        num_train_samples: Optional[int] = np.inf,
        batch_size: Optional[int] = np.inf,
        generator_use_initial: bool = True,
        time_scaling_coefficient: float = 10.0,
        return_dynamic_parameters: bool = False,
    ) -> tf.data.Dataset:
        """
        :params data_gen_rng: the random number generator
        :params x_0_range: the range of the initial condition
        :param t_range: the range of the time, NOTE that within the ODE, it has been hard coded to scale by 10 as done in Neural ODE Process paper
        :param num_context_range: the number of context points to be sampled
        :param num_timesteps: the number of total timesteps to be sampled, this will be uniformely spaced
        :param num_target_range: the number of target points to be sampled
        :param aux: dict the auxiliary datasets to be generated: {dataset_name: (num_aug_dynamic, num_aug_initial_cond)}
        :param generator: bool, whether to return a generator or a tf.data.Dataset
        :param dynamics_smp_num the number of dynamic systems to sample
        :param initial_condition_smp_num the number of initial conditions to sample within each dynamic systems
        :param num_train_samples: the number of training samples to be generated
        :param batch_size TODO
        :param generator_use_initial: bool, whether to use the initial condition as the context points
        """
        
        if generator is True:
            self._is_generator = True
        # note that this will make all the generated aux datasets the same since we do not change the rng,
        # this is ideal if we want to compare between different models as no need to store same data locally
        if aux is not None:
            data_gen_rng, aux_data_gen_rng = random.split(data_gen_rng, 2)
            aux_datasets = {}
            for key, (num_aug_dynamic, num_aug_initial_cond) in aux.items():
                x_samples, y_samples, _, params = self.sample_selkov_trajectory(
                    aux_data_gen_rng,
                    dynamic_sample_num=num_aug_dynamic,
                    initial_cond_sample_num=num_aug_initial_cond,
                    num_timesteps=num_timesteps,
                    initial_condition_range=tuple(
                        tuple(_x_0_range) for _x_0_range in x_0_range
                    ),
                    t_range=t_range,
                    a_range=a_range,
                    b_range = b_range,
                    use_initial=generator_use_initial,
                    num_context_range=num_context_range,
                    num_target_range=num_target_range,
                    time_scaling_coefficient = time_scaling_coefficient
                )
                # if generator == True:
                #     aux_datasets[key] = (x_samples, y_samples, context_mask, target_mask)
                # else:
                aux_datasets[key] = (x_samples, y_samples, params)
        else:
            aux_datasets = {}
        self.aux_datasets = aux_datasets

        # from matplotlib import pyplot as plt
        # plt.figure()
        # _, axs = plt.subplots(nrows=5, ncols=1, figsize=(3, 10))
        # for j in range(5):
        #     for i in range(num_aug_dynamic):
        #         axs[j].plot(x_samples[j, i], y_samples[j, i][..., 0], color='r')
        #         # axs[j].scatter(x_samples[j, i], y_samples[j, i][..., 0], s=10, color='r')
        #         axs[j].plot(x_samples[j, i], y_samples[j, i][..., 1], color='b')
        #         # axs[j].scatter(x_samples[j, i], y_samples[j, i][..., 1], s=10, color='b')
        # plt.xlabel('time')
        # plt.ylabel('population')
        # plt.suptitle('lotka volterra sample trajectories')
        # plt.savefig('SELKOV_samples.png', dpi=300)
        # raise ValueError

        def batch_training_data_generator(rng, batch_size):
            while True:
                x_samples, y_samples, rng, params = self.sample_selkov_trajectory(
                    rng,
                    dynamic_sample_num=dynamics_smp_num,
                    initial_cond_sample_num=initial_condition_smp_num,
                    num_timesteps=num_timesteps,
                    initial_condition_range=x_0_range,
                    t_range=t_range,
                    a_range=a_range,
                    b_range = b_range,
                    use_initial=generator_use_initial,
                    num_context_range=num_context_range,
                    num_target_range=num_target_range,
                    time_scaling_coefficient=time_scaling_coefficient
                )
                yield x_samples, y_samples, params

        if generator:
            data_gen = lambda: batch_training_data_generator(data_gen_rng, batch_size)
            train_data = tf.data.Dataset.from_generator(
                    data_gen, (np.float32, np.float32, np.float32)
                )  # # , np.bool_, np.bool_))
            return train_data
        else:
            x_samples, y_samples, rng, params = self.sample_selkov_trajectory(
                data_gen_rng,
                dynamic_sample_num=dynamics_smp_num,
                initial_cond_sample_num=initial_condition_smp_num,
                num_timesteps=num_timesteps,
                initial_condition_range=x_0_range,
                t_range=t_range,
                a_range=a_range,
                b_range = b_range,
                use_initial=generator_use_initial,
                num_context_range=num_context_range,
                num_target_range=num_target_range,
                time_scaling_coefficient=time_scaling_coefficient
            )
            train_data = tf.data.Dataset.from_tensor_slices((x_samples, y_samples, params))
            return train_data

    @staticmethod
    @partial(
        jit,
        static_argnames=(
            "dynamic_sample_num",
            "initial_cond_sample_num",
            "num_timesteps",
            "initial_condition_range",
            "t_range",
            "a_range",
            "b_range",
            "num_context_range",
            "num_target_range",
            "use_initial",
            "specified_times",
        ),
    )
    def sample_selkov_trajectory(
        rng: random.PRNGKey,
        dynamic_sample_num: int,
        initial_cond_sample_num: int,
        num_timesteps: int,
        initial_condition_range: List,
        t_range: List,
        a_range: List,
        b_range: List,
        num_context_range: Optional[List] = None,
        num_target_range: Optional[List] = None,
        use_initial: bool = False,
        specified_times: Optional[ArrayLike] = None,
        time_scaling_coefficient: float = 10.0,
    ):
        """
        generate the data from the lotka volterra system

        this function supports two approaches, one is used as a fixed dataset, the other one is use as a generator

        :param rng: the random number generator
        """
        rng, initial_rng, a_rng, b_rng = random.split(
            rng, 4
        )
        num_states = len(initial_condition_range[0])
        x0 = random.uniform(    
            initial_rng,
            shape=(dynamic_sample_num, initial_cond_sample_num, num_states),
            minval=np.asarray(initial_condition_range[0]),
            maxval=np.asarray(initial_condition_range[1]),
        )  # [dynamic_smp, ibatch_size, 1]

        b = random.uniform(b_rng, shape=(dynamic_sample_num, 1), minval=b_range[0], maxval=b_range[1])
        a = random.uniform(a_rng, shape=(dynamic_sample_num, 1), minval=a_range[0], maxval=a_range[1])
        a = np.repeat(np.expand_dims(a, axis=-2), initial_cond_sample_num, axis=-2)
        b = np.repeat(np.expand_dims(b, axis=-2), initial_cond_sample_num, axis=-2)
        
        def dynamics(t, x, args):
            _x, _y = np.split(x, 2, axis=-1)
            _dx = -_x + a * _y + (_x ** 2) * _y
            _dy = b - a * _y - (_x ** 2) * _y
            return np.concatenate([_dx, _dy], axis=-1)

        if specified_times is None:
            t_samples = np.linspace(t_range[0], t_range[1], num_timesteps)
        else:
            t_samples = specified_times
        raw_t_samples = (
            t_samples * time_scaling_coefficient
        )  # note this is used in NODEP paper that it scaled the time by 10
        raw_t_range = [t_range[0], time_scaling_coefficient * (t_range[1] - t_range[0])]
        rtol, atol = 1e-7, 1e-9
        while True:
            try:
                # solve ode
                traj = diffeqsolve(
                    ODETerm(dynamics),
                    t0=raw_t_range[0],
                    t1=raw_t_range[1],
                    dt0=None,
                    stepsize_controller=PIDController(rtol=rtol, atol=atol),
                    y0=x0,
                    solver=Dopri5(),
                    max_steps=5000,
                    throw=False,
                    saveat=SaveAt(ts=raw_t_samples),
                ).ys
                break
            except Exception as e:
                print('Data generator encountered ODE solving issue, using a larger rtol and atl')
                rtol *= 10
                atol *= 10
        traj = rearrange(traj, "times batch traj_size dim -> batch traj_size times dim")

        t_samples = np.repeat(
            np.repeat(t_samples[None, ...], repeats=initial_cond_sample_num, axis=0)[
                None, ...
            ],
            repeats=dynamic_sample_num,
            axis=0,
        )[
            ..., None
        ]  # [dynamic_sample_num, initial_cond_sample_num, timesteps, 1]
        context_mask = target_mask = None

        return t_samples, traj, rng, np.concatenate([a, b], axis=-1)

    def get_data_input_scaler(self, inputs):
        return inputs

    def get_data_inverse_scaler(self, inputs):
        return inputs

    def get_aux_datasets(self):
        return self.aux_datasets

    @property
    def is_generator(self):
        return self._is_generator


class SIR_Unormalized_ODE(TFDataset):
    """
    Implementation of the unnormalized version of SIR System
    """

    def generate_tf_dataset(
            self,
            data_gen_rng: random.PRNGKey,
            S_range: List,
            I_range: List,
            beta_range: List,
            gamma_range: List,
            # R0_range: List,
            t_range: Optional[List] = [0, 1.0],
            num_context_range: Optional[List] = None,
            num_timesteps: Optional[int] = None,
            num_target_range: Optional[int] = None,
            aux: Optional[dict] = None,
            generator: bool = True,
            dynamics_smp_num: int = 20,
            initial_condition_smp_num: int = 20,
            num_train_samples: Optional[int] = np.inf,
            batch_size: Optional[int] = np.inf,
            generator_use_initial: bool = True,
            time_scaling_coefficient: float = 1.0,
            ):
        
        if generator is True:
            self._is_generator = True
        # note that this will make all the generated aux datasets the same since we do not change the rng,
        # this is ideal if we want to compare between different models as no need to store same data locally
        if aux is not None:
            data_gen_rng, aux_data_gen_rng = random.split(data_gen_rng, 2)
            aux_datasets = {}
            for key, (num_aug_dynamic, num_aug_initial_cond) in aux.items():
                x_samples, y_samples, _, params = self.sample_sir_trajectory(
                    aux_data_gen_rng,
                    dynamic_sample_num=num_aug_dynamic,
                    initial_cond_sample_num=num_aug_initial_cond,
                    num_timesteps=num_timesteps,
                    t_range=t_range,
                    # R0_range = R0_range,
                    beta_range=beta_range,
                    gamma_range=gamma_range,
                    s_range=S_range,
                    i_range=I_range,
                    use_initial=generator_use_initial,
                    num_context_range=num_context_range,
                    num_target_range=num_target_range,
                    time_scaling_coefficient = time_scaling_coefficient
                )
                # if generator == True:
                #     aux_datasets[key] = (x_samples, y_samples, context_mask, target_mask)
                # else:
                aux_datasets[key] = (x_samples, y_samples, params)
        else:
            aux_datasets = {}
        self.aux_datasets = aux_datasets

        # plot sir curves
        

        # Assume x_samples and y_samples are your data
        # Create a figure with 10 subplots arranged in a 5x2 grid
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(5, 2, figsize=(10, 20))

        # # Reshape axs to a 1D array to make it easier to iterate over
        # axs = axs.reshape(-1)

        # # Iterate over the first 10 subplots and the data
        # for ax, x, y in zip(axs, x_samples, y_samples):
        #     # Squeeze the last dimension from x and y
        #     x = x.squeeze(-1)

        #     for xi, yi in zip(x[:10], y[:10]):
        #         ax.plot(xi, yi[..., 0], color='r')  # S
        #         ax.plot(xi, yi[..., 1], color='g')  # I
        #         ax.plot(xi, yi[..., 2], color='b')  # R
        #     # Plot the curves in this subplot

        # # Show the figure
        # plt.suptitle('SIR model sample trajectories')
        # plt.savefig('sir_model.png', dpi=300)


        def batch_training_data_generator(rng, batch_size):
            while True:
                x_samples, y_samples, rng, params = self.sample_sir_trajectory(
                    rng,
                    dynamic_sample_num=dynamics_smp_num,
                    initial_cond_sample_num=initial_condition_smp_num,
                    num_timesteps=num_timesteps,
                    t_range=t_range,
                    beta_range=beta_range,
                    gamma_range=gamma_range,
                    # R0_range = R0_range,
                    s_range=S_range,
                    i_range=I_range,
                    use_initial=generator_use_initial,
                    num_context_range=num_context_range,
                    num_target_range=num_target_range,
                    time_scaling_coefficient = time_scaling_coefficient
                )
                yield x_samples, y_samples, params

        if generator:
            data_gen = lambda: batch_training_data_generator(data_gen_rng, batch_size)
            train_data = tf.data.Dataset.from_generator(
                    data_gen, (np.float32, np.float32, np.float32)
                )  # # , np.bool_, np.bool_))
            return train_data
        else:
            x_samples, y_samples, rng, params = self.sample_sir_trajectory(
                data_gen_rng,
                dynamic_sample_num=dynamics_smp_num,
                initial_cond_sample_num=initial_condition_smp_num,
                num_timesteps=num_timesteps,
                t_range=t_range,
                # R0_range = R0_range,
                beta_range=beta_range,
                gamma_range=gamma_range,
                s_range=S_range,
                i_range=I_range,
                use_initial=generator_use_initial,
                num_context_range=num_context_range,
                num_target_range=num_target_range,
                time_scaling_coefficient = time_scaling_coefficient
            )
            train_data = tf.data.Dataset.from_tensor_slices((x_samples, y_samples, params))
            return train_data


    @staticmethod
    @partial(
        jit,
        static_argnames=(
            "dynamic_sample_num",
            "initial_cond_sample_num",
            "num_timesteps",
            "t_range",
            "beta_range",
            "gamma_range",
            "s_range", 
            "i_range",
            "num_context_range",
            "num_target_range",
            "use_initial",
            "specified_times",
            "time_scaling_coefficient"
        ),
    )
    def sample_sir_trajectory(
            rng: random.PRNGKey,
            dynamic_sample_num: int,
            initial_cond_sample_num: int,
            num_timesteps: int,
            t_range: List,
            # R0_range: List,
            beta_range: List,
            gamma_range: List,
            s_range: List,
            i_range: List,
            num_context_range: Optional[List] = None,
            num_target_range: Optional[List] = None,
            use_initial: bool = False,
            specified_times: Optional[ArrayLike] = None,
            time_scaling_coefficient = 1.0,
    ):
        rng, initial_rng, beta_rng, gamma_rng  = random.split(rng, 4) # 

        # sample initial condition using Dirichlet distribution
        S = random.uniform(initial_rng, shape=(dynamic_sample_num, initial_cond_sample_num), minval=s_range[0], maxval=s_range[1])
        I = random.uniform(initial_rng, shape=(dynamic_sample_num, initial_cond_sample_num), minval=i_range[0], maxval=i_range[1]) # 1.0 - S
        R = np.zeros_like(S)
        x0 = np.stack([S, I, R], axis=-1)

        beta = np.repeat(random.uniform(
            beta_rng,
            shape=(dynamic_sample_num,),
            minval=beta_range[0],
            maxval=beta_range[1],
        )[..., None], initial_cond_sample_num, axis=-1)  # Contact rate.
# 
        gamma = np.repeat(random.uniform(
            gamma_rng,
            shape=(dynamic_sample_num,),
            minval=gamma_range[0],
            maxval=gamma_range[1],
        )[..., None], initial_cond_sample_num, axis=-1)  # Mean recovery rate.

        def sir_dynamics(t, y, args):
            normed_S, normed_I, _ = y[..., 0], y[..., 1], y[..., 2] 
            dSdt = -beta * normed_S * normed_I 
            dIdt = beta * normed_S * normed_I  - gamma * normed_I
            dRdt = gamma * normed_I
            return np.stack([dSdt, dIdt, dRdt], axis=-1)
        
        if specified_times is None:
            t_samples = np.linspace(t_range[0], t_range[1], num_timesteps)
        else:
            t_samples = specified_times
        raw_t_samples = (
            t_samples * time_scaling_coefficient
        )  # note this is used in NODEP paper that it scaled the time by 10
        raw_t_range = [t_range[0], time_scaling_coefficient * (t_range[1] - t_range[0])]

        # solve ode
        traj = diffeqsolve(
            ODETerm(sir_dynamics),
            t0=raw_t_range[0],
            t1=raw_t_range[1],
            dt0=None,
            stepsize_controller=PIDController(rtol=1e-9, atol=1e-9),
            y0=x0,
            solver=Dopri5(),
            max_steps=5000,
            throw=False,
            saveat=SaveAt(ts=raw_t_samples),
        ).ys
        traj = rearrange(traj, "times batch traj_size dim -> batch traj_size times dim")

        t_samples = np.repeat(
            np.repeat(t_samples[None, ...], repeats=initial_cond_sample_num, axis=0)[
                None, ...
            ],
            repeats=dynamic_sample_num,
            axis=0,
        )[
            ..., None
        ]  # [dynamic_sample_num, initial_cond_sample_num, timesteps, 1]

        return t_samples, traj, rng, np.concatenate([beta[..., None], gamma[..., None]], axis=-1)

    def get_data_input_scaler(self, inputs):
        return inputs

    def get_data_inverse_scaler(self, inputs):
        return inputs

    def get_aux_datasets(self):
        return self.aux_datasets
    

class SIRD(TFDataset):
    def generate_tf_dataset(
            self,
            data_gen_rng: random.PRNGKey,
            beta_range,
            gamma_range,
            mu_range,
            s_range,
            i_range, 
            t_range: Optional[List] = [0, 1.0],
            num_context_range: Optional[List] = None,
            num_timesteps: Optional[int] = None,
            num_target_range: Optional[int] = None,
            aux: Optional[dict] = None,
            generator: bool = True,
            dynamics_smp_num: int = 20,
            initial_condition_smp_num: int = 20,
            num_train_samples: Optional[int] = np.inf,
            batch_size: Optional[int] = np.inf,
            generator_use_initial: bool = True,
            time_scaling_coefficient: float = 1.0,
            return_dynamic_parameters: bool = False,
            ):
        
        if generator is True:
            self._is_generator = True
        # note that this will make all the generated aux datasets the same since we do not change the rng,
        # this is ideal if we want to compare between different models as no need to store same data locally
        if aux is not None:
            data_gen_rng, aux_data_gen_rng = random.split(data_gen_rng, 2)
            aux_datasets = {}
            for key, (num_aug_dynamic, num_aug_initial_cond) in aux.items():
                x_samples, y_samples, _, params = self.sample_sird_trajectory(
                    aux_data_gen_rng,
                    dynamic_sample_num=num_aug_dynamic,
                    initial_cond_sample_num=num_aug_initial_cond,
                    num_timesteps=num_timesteps,
                    t_range=t_range,
                    beta_range = beta_range,
                    gamma_range = gamma_range,
                    mu_range = mu_range,
                    s_range = s_range,
                    i_range=i_range,
                    use_initial=generator_use_initial,
                    num_context_range=num_context_range,
                    num_target_range=num_target_range,
                    time_scaling_coefficient = time_scaling_coefficient
                )
                # if generator == True:
                #     aux_datasets[key] = (x_samples, y_samples, context_mask, target_mask)
                # else:
                aux_datasets[key] = (x_samples, y_samples, params)
        else:
            aux_datasets = {}
        self.aux_datasets = aux_datasets

        # plot sir curves
        

        # Assume x_samples and y_samples are your data
        # Create a figure with 10 subplots arranged in a 5x2 grid
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(5, 2, figsize=(10, 20))

        # # Reshape axs to a 1D array to make it easier to iterate over
        # axs = axs.reshape(-1)

        # # Iterate over the first 10 subplots and the data
        # for ax, x, y in zip(axs, x_samples, y_samples):
        #     # Squeeze the last dimension from x and y
        #     x = x.squeeze(-1)

        #     for xi, yi in zip(x[:10], y[:10]):
        #         ax.plot(xi, yi[..., 0], color='r')  # S
        #         ax.plot(xi, yi[..., 1], color='g')  # E
        #         ax.plot(xi, yi[..., 2], color='b')  # I
        #         ax.plot(xi, yi[..., 3], color='k')  # R
        #     # Plot the curves in this subplot

        # # Show the figure
        # plt.suptitle('SIR model sample trajectories')
        # plt.savefig('seir_model.png', dpi=300)


        def batch_training_data_generator(rng, batch_size):
            while True:
                x_samples, y_samples, rng, params = self.sample_sird_trajectory(
                    rng,
                    dynamic_sample_num=dynamics_smp_num,
                    initial_cond_sample_num=initial_condition_smp_num,
                    num_timesteps=num_timesteps,
                    t_range=t_range,
                    beta_range = beta_range,
                    gamma_range = gamma_range,
                    mu_range = mu_range,
                    s_range = s_range,
                    i_range=i_range,
                    use_initial=generator_use_initial,
                    num_context_range=num_context_range,
                    num_target_range=num_target_range,
                    time_scaling_coefficient = time_scaling_coefficient
                )
                yield x_samples, y_samples, params

        if generator:
            data_gen = lambda: batch_training_data_generator(data_gen_rng, batch_size)
            train_data = tf.data.Dataset.from_generator(
                    data_gen, (np.float32, np.float32, np.float32)
                )
            return train_data
        else:
            x_samples, y_samples, rng, params = self.sample_sird_trajectory(
                data_gen_rng,
                dynamic_sample_num=dynamics_smp_num,
                initial_cond_sample_num=initial_condition_smp_num,
                num_timesteps=num_timesteps,
                t_range=t_range,
                beta_range = beta_range,
                gamma_range = gamma_range,
                mu_range = mu_range,
                s_range = s_range,
                i_range=i_range,
                use_initial=generator_use_initial,
                num_context_range=num_context_range,
                num_target_range=num_target_range,
                time_scaling_coefficient = time_scaling_coefficient
            )
            train_data = tf.data.Dataset.from_tensor_slices((x_samples, y_samples, params))
            return train_data


    @staticmethod
    @partial(
        jit,
        static_argnames=(
            "dynamic_sample_num",
            "initial_cond_sample_num",
            "num_timesteps",
            "t_range",
            "beta_range", 
            "gamma_range", 
            "mu_range",
            "s_range",	
            "i_range", 
            "num_context_range",
            "num_target_range",
            "use_initial",
            "specified_times",
        ),
    )
    def sample_sird_trajectory(
            rng: random.PRNGKey,
            dynamic_sample_num: int,
            initial_cond_sample_num: int,
            num_timesteps: int,
            t_range: List,
            beta_range: List,
            gamma_range: List,
            mu_range: List, 
            s_range: List,
            i_range: List,
            num_context_range: Optional[List] = None,
            num_target_range: Optional[List] = None,
            use_initial: bool = False,
            specified_times: Optional[ArrayLike] = None,
            time_scaling_coefficient = 10.0
    ):
        rng, initial_rng, beta_rng, gamma_rng, mu_rng = random.split(rng, 5) # beta_rng, gamma_rng 

        # sample initial condition using Dirichlet distribution
        # x0 = random.dirichlet(initial_rng, alpha=np.array([1/3, 1/3, 1/3]), shape=(dynamic_sample_num, initial_cond_sample_num))
        S = random.uniform(initial_rng, shape=(dynamic_sample_num, initial_cond_sample_num), minval=s_range[0], maxval=s_range[1])
        I = random.uniform(initial_rng, shape=(dynamic_sample_num, initial_cond_sample_num), minval=i_range[0], maxval=i_range[1]) # 1.0 - S
        R = np.zeros_like(S)
        D = np.zeros_like(S)
        x0 = np.stack([S, I, R, D], axis=-1)

        beta = np.repeat(
            random.uniform(
                beta_rng,
                shape=(dynamic_sample_num,),
                minval=beta_range[0],
                maxval=beta_range[1],
            )[..., None],
            initial_cond_sample_num,
            axis=-1,
        )  # [dynamic_smp, ibatch_size, 1]

        gamma = np.repeat(
            random.uniform(
                gamma_rng,
                shape=(dynamic_sample_num,),
                minval=gamma_range[0],
                maxval=gamma_range[1],
            )[..., None],
            initial_cond_sample_num,
            axis=-1,
        )  # [dynamic_smp, ibatch_size, 1]

        mu = np.repeat(
            random.uniform(
                mu_rng,
                shape=(dynamic_sample_num,),
                minval=mu_range[0],
                maxval=mu_range[1],
            )[..., None],
            initial_cond_sample_num,
            axis=-1,
        )  # [dynamic_smp, ibatch_size, 1]


        def sird_dynamics(t, y, args):
            normed_S, normed_I, normed_R, normed_D = y[..., 0], y[..., 1], y[..., 2], y[..., 3] 
            dSdt = -beta * normed_S * normed_I
            dIdt = beta * normed_S * normed_I - gamma * normed_I - mu * normed_I
            dRdt = gamma * normed_I
            dDdt = mu * normed_I
            return np.stack([dSdt, dIdt, dRdt, dDdt], axis=-1)

        if specified_times is None:
            t_samples = np.linspace(t_range[0], t_range[1], num_timesteps)
        else:
            t_samples = specified_times
        raw_t_samples = (
            t_samples * time_scaling_coefficient
        )  # note this is used in NODEP paper that it scaled the time by 10
        raw_t_range = [t_range[0], time_scaling_coefficient * (t_range[1] - t_range[0])]

        # solve ode
        traj = diffeqsolve(
            ODETerm(sird_dynamics),
            t0=raw_t_range[0],
            t1=raw_t_range[1],
            dt0=None,
            stepsize_controller=PIDController(rtol=1e-7, atol=1e-9),
            y0=x0,
            solver=Dopri5(),
            max_steps=5000,
            throw=False,
            saveat=SaveAt(ts=raw_t_samples),
        ).ys
        traj = rearrange(traj, "times batch traj_size dim -> batch traj_size times dim")

        t_samples = np.repeat(
            np.repeat(t_samples[None, ...], repeats=initial_cond_sample_num, axis=0)[
                None, ...
            ],
            repeats=dynamic_sample_num,
            axis=0,
        )[
            ..., None
        ]  # [dynamic_sample_num, initial_cond_sample_num, timesteps, 1]

        return t_samples, traj, rng, np.concatenate([beta[..., None], gamma[..., None], mu[..., None]], axis=-1)

    def get_data_input_scaler(self, inputs):
        return inputs

    def get_data_inverse_scaler(self, inputs):
        return inputs

    def get_aux_datasets(self):
        return self.aux_datasets


class SingleTrajectoryDataset:
    def __init__(
        self,
        state_dim: int,
        times: Optional[ArrayLike],
        observations: Optional[ArrayLike],
        initial_cond: Optional[ArrayLike],
    ) -> None:
        """
        Dataset coming from single trajectories, not that the data is enforced to store in order since the
        model is order invariant

        :params state_dim: the initial time of the trajectory
        :params times: [total_timesteps]
        :params observations: [total_timesteps, state_dim] the measurements at different times
        :param initial_cond: [state_dim] the initial condition of the trajectory
        """
        assert (times is None and observations is None) or (
            times is not None and observations is not None
        ), ValueError("times and observations should be both None or not None")
        if times is not None and observations is not None:
            self.times = times
            self.states = observations
        else:
            self.times = np.zeros(shape=(0,))
            self.states = np.zeros(shape=(0, state_dim))
        self.init_cond = initial_cond

    def append(self, times: ArrayLike, observations: ArrayLike):
        """
        Add new time and observations in the same trajectory
        """
        self.times = np.concatenate([self.times, times], axis=0)
        self.states = np.concatenate([self.states, observations], axis=0)


class NP_Dataset:
    """
    Optimization data structure
    """

    def __init__(self, trajectory_dicts: dict, initial_cond_mapping = lambda x: x) -> None:
        """
        Note that the trajectory_dicts is the type of
        {'traj_id': ((initial_condition), (times, obs))}
        :params initial_cond_mapping: a function that maps the initial condition to the model input, this is used especially when 
        the decision initial condition has lower dim than the actual model state dim
        """
        self.traj_dicts = trajectory_dicts
        self.state_dim = self.traj_dicts[list(self.traj_dicts.keys())[0]].states.shape[
            -1
        ]
        self.initial_cond_mapping = initial_cond_mapping

    def append_new_traj(self, key: str, traj_inst: SingleTrajectoryDataset):
        self.traj_dicts[key] = traj_inst

    def append_obs_within_traj(
        self, key: str, times: List[ArrayLike], observations: List[ArrayLike]
    ):
        """
        Append observations in certain trajectories
        """
        self.traj_dicts[key].append(times, observations)

    # TODO:
    def formalize_training_data_sanodep(self) -> List[ArrayLike]:
        """
        Convert the multi trajectory stored dict data into the format that can be used in the SANODEP model
        """
        self._pad_and_concatenate_for_test_usage()
        return self.times, self.states, self.masks

    # 2024/08/05 I don't think we should jit this, as said, when init_cond and star_time is not changing, it will use 
    # cached attributes, no matter whether self.states has updated or not (as it will use cached self.states)
    # @partial(jit, static_argnums=(0,))
    def formalize_training_data_with_pred_init_cond(
        self, init_cond: ArrayLike, start_time: ArrayLike
    ) -> List[ArrayLike]:
        """
        when predicting at a new initial condition, augment the initial condition into the model context dataset.
        This has been achived by having a new trajectory with the initial condition and the start time.
        Note that to eval this method, formalize_training_data_sanodep need to be evaled first to initailize times, states and masks
        """
        max_time_length = self.times.shape[1]
        time = start_time
        padded_test_times = np.pad(
            time,
            ((0, 0), (0, max_time_length - time.shape[-1])),
            "constant",
            constant_values=0,
        )
        padded_test_states = np.pad(
            self.initial_cond_mapping(init_cond),
            ((0, max_time_length - len(init_cond)), (0, 0)),
            "constant",
            constant_values=0,
        )
        mask = np.ones_like(time)
        padded_test_mask = np.pad(
            mask,
            ((0, 0), (0, max_time_length - time.shape[-1])),
            "constant",
            constant_values=0,
        )
        return (
            np.concatenate([padded_test_times, self.times], axis=0),
            np.concatenate([padded_test_states[None, ...], self.states], axis=0),
            np.concatenate([padded_test_mask, self.masks], axis=0),
        )

    def _pad_and_concatenate_for_test_usage(self):
        """
        Pad and concatenate times and states from multiple SingleTrajectoryDataset instances

        Note that, this is only for model test usage, which will only use the first trajectory's timesteps
        in diffeqsolve, and hence, we don't have to do sth like finding a unified timestep for all the trajectories
        """
        max_len = max(traj.times.shape[0] for traj in self.traj_dicts.values())

        padded_times = []
        padded_states = []
        masks = []

        for traj in self.traj_dicts.values():
            pad_len = max_len - traj.times.shape[0]

            padded_time = np.pad(
                traj.times, (0, pad_len), "constant", constant_values=0
            )
            padded_state = np.pad(
                traj.states, ((0, pad_len), (0, 0)), "constant", constant_values=0
            )
            mask = np.pad(
                np.ones_like(traj.times), (0, pad_len), "constant", constant_values=0
            )
            mask = mask.astype(bool)

            padded_times.append(padded_time)
            padded_states.append(padded_state)
            masks.append(mask)

        self.times = np.stack(padded_times, axis=0)
        self.states = np.stack(padded_states, axis=0)
        self.masks = np.stack(masks, axis=0)
        return self.times, self.states, self.masks

    def formalize_training_data_for_GPJax(self) -> List[ArrayLike]:
        gp_inputs = []
        gp_states = []

        for traj in self.traj_dicts.values():
            # [total_timesteps, state_dim + 1]
            gp_inputs.append(
                np.concatenate(
                    [
                        np.repeat(
                            traj.init_cond[None, ...], traj.times.shape[0], axis=0
                        ),
                        traj.times[..., None],
                    ],
                    axis=-1,
                )
            )
            gp_states.append(traj.states)
        return np.concatenate(gp_inputs, axis=0), np.concatenate(gp_states, axis=0)

    def formalize_training_data_for_trieste(self, dtype) -> Dataset:
        from trieste.data import Dataset
        import tensorflow as tf

        gp_inputs = []
        gp_states = []

        for traj in self.traj_dicts.values():
            # [total_timesteps, state_dim + 1]
            gp_inputs.append(
                np.concatenate(
                    [
                        np.repeat(
                            (traj.init_cond[None, ...]), traj.times.shape[0], axis=0
                        ),
                        traj.times[..., None],
                    ],
                    axis=-1,
                )
            )
            gp_states.append(traj.states)
        # np.concatenate(gp_inputs, axis=0), np.concatenate(gp_states, axis=0)
        return Dataset(
            tf.cast(np.concatenate(gp_inputs, axis=0), dtype=dtype),
            tf.cast(np.concatenate(gp_states, axis=0), dtype=dtype),
        )

        # # TODO: 遇到了一个问题: 这个
        # def formalize_training_data_with_predict_data(self, init_cond, time) -> List[ArrayLike]:
        #     max_time_length = self.times.shape[1]
        #     padded_times = np.pad(time, (0, max_time_length - len(time)), 'constant', constant_values=0)
        #     padded_states = np.pad(init_cond, ((0, max_time_length - len(init_cond)), (0, 0)), 'constant', constant_values=0)
        #     mask = np.ones_like(time)
        #     mask = np.pad(mask, (0, max_time_length - len(time)), 'constant', constant_values=0)
        #     return np.stack([padded_times]), np.stack([padded_states]), np.stack([mask])
        #
        # # this is not jittable because of np.unique
        # @staticmethod
        # def create_unified_timeline(time_series_list):
        # Combine all timesteps and find unique ones
        all_timesteps = np.concatenate(time_series_list)
        unique_timesteps = np.unique(all_timesteps)
        return np.sort(unique_timesteps)
        #
        # @staticmethod
        # # @jit
        # def align_series_to_timeline(ts, obs, unified_timeline):
        # Initialize padded observations with NaNs or an appropriate placeholder
        padded_obs = np.full_like(
            np.zeros(shape=(unified_timeline.shape[0], obs.shape[-1])),
            fill_value=np.nan,
            dtype=obs.dtype,
        )

        # Create a mask array with 0s
        # mask = np.zeros_like(unified_timeline, dtype=np.bool_)

        # Find the indices of the ts in the unified timeline
        bool_mask = np.isin(
            unified_timeline, ts
        )  # np.repeat(np.isin(unified_timeline, ts)[..., None], obs.shape[-1], axis=-1)
        # padded_obs = np.where(bool_mask, padded_obs, padded_obs)
        indices = np.where(bool_mask)[0]
        # Use advanced indexing to fill in the observations and mask
        padded_obs = padded_obs.at[indices, :].set(obs)
        # mask = mask.at[indices].set(np.ones_like(obs, dtype=np.bool_))

        return padded_obs, bool_mask

        # For each timestep in the current series, find its position in the unified timeline and fill in the observation
        for i, timestep in enumerate(ts):
            index = np.where(unified_timeline == timestep)[0]
            if index.size > 0:  # If the timestep exists in the unified timeline
                padded_obs = padded_obs.at[index].set(obs[i])
                mask = mask.at[index].set(1)

        return padded_obs, mask


