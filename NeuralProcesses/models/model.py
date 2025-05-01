from flax.linen import Module, Dense, Sequential, tanh
from .encoder import Encoder, MuSigmaEncoder, MuSigmaFixedLikelihoodEncoder
from .decoder import (
    HeteroscedasticNODEPDecoder,
    FixedLikelihoodNODEPDecoder,
    HeteroscedasticDecoder, 
    FixedLikelihoodDecoder, 
    PI_HeteroscedasticNODEPDecoder,
)

from flax.linen import Module, Dense, Sequential, tanh
from .utils.transformation import IdentityTransform
from abc import abstractmethod
from typing import Optional
from jax import numpy as np
from jax import random, vmap
from jax.typing import ArrayLike

from abc import ABC
from diffrax import diffeqsolve, ODETerm, PIDController
import diffrax
from einops import rearrange

from tensorflow_probability.substrates import jax as tfp

distributions = tfp.distributions


class __AbstractSingleTimeSeriesInputModel(ABC):
    """
    Model that takes single timeseries as input
    The input shape for __call__ method will be
    [timesteps, state_num]
    Note that the batch axis of data is handled by vmap outside model.__call__ method
    """


class __AbstractMultiTimeSeriesInputModel(ABC):
    """
    Model that takes multiple timeseries as input
    The input shape for __call__ method will typically be
    [timeseries_number, timesteps, state_num]
    Note that the batch axis of data is handled by vmap outside model.__call__ method
    """

class _NeuralProcess(Module):
    x_dim: int
    y_dim: int
    r_dim: int
    z_dim: int
    encoder_h_dim: int
    decoder_h_dim: int
    variational_distribution_std_lower_bound: float = 0.1
    heteroscedastic_noise: bool = True
    fixed_likelihood_noise: bool = False
    decoder_std_lower_bound: Optional[float] = 1E-4
    decoder_std: Optional[float] = 1E-3
    """
    Implements Neural Process for functions of arbitrary dimensions.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.
    y_dim : int
        Dimension of y values.
    r_dim : int
        Dimension of output representation r.
    z_dim : int
        Dimension of latent variable z.
    encoder_h_dim : int
        Dimension of hidden layer in encoder
    decoder_h_dim : int
        Dimension of hidden layer in decoder
    variational_distribution_std_lower_bound: float
        Minimum predictive standard deviation of the amortized variational posterior of the latent variable
    heteroscedastic_noise: bool 
        whether the decoder will output a heteroscedastic noise (variance) for each data
    fixed_likelihood_noise: bool 
        if set to true, the decoder will output a fixed likelihod variance (optionally) specified by users
    decoder_std_lower_bound: float
        Minimum predictive standard deviation of the decoder model
    """
    def setup(self):
        # Initialize networks
        self._xy_to_r = Encoder(self.encoder_h_dim, self.r_dim)
        self._r_to_mu_sigma = MuSigmaEncoder(self.r_dim, self.z_dim, std_lower_bound=self.variational_distribution_std_lower_bound)
        if self.heteroscedastic_noise and not self.fixed_likelihood_noise:
            self._xz_to_y = HeteroscedasticDecoder(self.x_dim, self.z_dim, self.decoder_h_dim, self.y_dim, std_lower_bound=self.decoder_std_lower_bound)
        elif self.fixed_likelihood_noise:
            self._xz_to_y = FixedLikelihoodDecoder(self.x_dim, self.z_dim, self.decoder_h_dim, self.y_dim, std=self.decoder_std)
        else:
            raise NotImplementedError('Decoder Type Not Understood!')

    def aggregate(self, r_i, mask: Optional[ArrayLike] = None):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : [num_points, r_dim]
        """
        if mask is None:
            return np.mean(r_i, axis=-2)
        else:
            return np.sum(r_i * mask[..., None], axis=-2) / np.sum(mask, axis=-1, keepdims=True)

    def xy_to_z_dist(self, x, y, mask: Optional[ArrayLike] = None):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        x : [batch_size, num_points, x_dim]
        y : [batch_size, num_points, y_dim]
        """
        # Encode each point into a representation r_i
        r_i = self._xy_to_r(np.concatenate([x, y], axis=-1)) # [num_points, r_dim]
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i, mask)  # [r_dim]
        # Return parameters of distribution
        return self._r_to_mu_sigma(r)  # [r_dim], [r_dim]

    def xz_to_y(self, x, z):
        return self._xz_to_y(x, z)


    def __call__(self, x_context, y_context, x_target, sample_rng: random.PRNGKey, 
                 sample_size: int, y_target=None, context_mask = None, 
                 target_mask = None, training=True):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.

        Parameters
        ----------
        x_context : [num_context, x_dim]. Note that x_context is a subset of x_target.
        y_context : [num_context, y_dim]
        x_target : [num_target, x_dim]
        y_target : [num_target, y_dim]. Only used during training.
        sample_rng: 
        sample_size: int
        training: future drop out usage

        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        if training:
            assert y_target is not None
            # Encode target and context (context needs to be encoded to calculate kl term)
            mu_context, sigma_context = self.xy_to_z_dist(x_context, y_context, context_mask)
            mu_tgt, sigma_tgt = self.xy_to_z_dist(x_target, y_target, target_mask)
            # Sample from encoded distribution using reparameterization trick
            # Get parameters of output distribution
            z_sample = np.expand_dims(mu_tgt, axis=0) + \
                random.normal(sample_rng, shape=(sample_size, self.z_dim)) * sigma_tgt
            y_pred_mu, y_pred_sigma = self._xz_to_y(x_target, z_sample) # [sample_size, num_all, y_dim]
            # p_y_pred = distributions.Normal(y_pred_mu, y_pred_sigma) # [sample_size, num_target, y_dim]

            return y_pred_mu, y_pred_sigma, mu_context, sigma_context, mu_tgt, sigma_tgt # q_context, q_tgt
        else:
            # At testing time, encode only context
            mu_context, sigma_context = self.xy_to_z_dist(x_context, y_context, context_mask)
            # Sample from distribution based on context
            q_context = distributions.MultivariateNormalDiag(mu_context, sigma_context)
            z_sample = q_context.sample(sample_size, seed=sample_rng)
            # Predict target points based on context
            y_pred_mu, y_pred_sigma = self._xz_to_y(x_target, z_sample)

            return y_pred_mu, y_pred_sigma


class NeuralProcessAcceptSystemData(_NeuralProcess, __AbstractMultiTimeSeriesInputModel):
    t0: float = np.inf
    t1: float = np.inf
    """
    An extended Neural Process model, the only difference of this model with the Neural Process 
    is that it accepts multi trajectories data from the same dynamic system, so it utilize the same 
    training data flow as SANODEP, which can be serve as a good comparison of NP vs SANODEP (
        see if the inductive bias of SANODEP can make a difference)

    Because this is mainly used for comparison with SANODEP, we modify its input structure to be similar as 
    in SANODEP
    """   

    def aggregate(self, r_i, mask: Optional[ArrayLike] = None):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : [traj_size num_points, r_dim]
        """
        flatten_r_i = rearrange(
            r_i, "traj_size num_points r_dim -> (traj_size num_points) r_dim"
        )  # rearrange the embedded context to be the standard NP data structure
        flatten_mask = rearrange(
            mask,
            "traj_size1 traj_size2 num_points -> traj_size1 (traj_size2 num_points)",
        )  # [traj_size, traj_size * num_points]
        # take the mean of the last axis across thoses, the usage of this looks bit weird implementation
        # is only to ensure that the average is operated across those masked True elements
        r_system = np.sum(flatten_r_i * flatten_mask[..., None], axis=-2) / np.maximum(
            np.sum(flatten_mask, axis=-1, keepdims=True), 1
        )
    
        return r_system
        
    def xy_to_z_dist(self, x, y, mask: Optional[ArrayLike] = None):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        x : [traj_size, num_points, x_dim]
        y : [traj_size, num_points, y_dim]

        mask: [traj_size, traj_size, num_points]
        """
        # Encode each point into a representation r_i
        r_i = self._xy_to_r(np.concatenate([x, y], axis=-1)) # [traj_size, num_points, r_dim]
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i, mask)  # [traj_size, r_dim]
        # Return parameters of distribution
        return self._r_to_mu_sigma(r)  # [traj_size, r_dim], [traj_size, r_dim]
    
    def xz_to_y(self, x, z):
        """
        :params x [traj_size, timesteps, x_dim + 1]
        :params z [traj_size, sample_size, z_dim]

        return [traj_size, timesteps, sample_size, y_dim]
        """
        pred_mean, pred_std = vmap(self._xz_to_y)(x, z) # [traj_size, sample_size, timesteps, y_dim]
        return pred_mean, pred_std

    def __call__(
        self,
        t_context,
        x_context,
        t_target,
        sample_rng,
        target_initial_cond_mask,
        ctx_mask_with_new_traj_obs,
        ctx_mask_with_new_traj_target_mask,
        sample_size: int,
        x_target,
        training,
        t0=None,
        t1=None,
        solver="Dopri5",
        **kwargs
    ):
        """
        t_context : [traj_size, timesteps]
        x_context : [traj_size, timesteps, x_dim]. Note that x_context is a subset of x_target.
        t_target : [traj_size, timesteps']
        x_target : Optional [traj_size, timesteps', x_dim]. Only used during training.
        ctx_mask_with_new_traj_obs: [traj_size, traj_size, num_points], this could represent problem setting 1
            and problem setting 2
        ctx_mask_with_new_traj_target_mask: [traj_size, traj_size, num_points]
        """
        if training:
            # reformulate the dynamic system data to be the same input formate for Neural Processes
            t_context = t_context[..., None] # [traj_size, timesteps, 1]
            _ctx_helper_input_initial_cond = np.repeat(x_context[:, :1, :], axis=-2, repeats=t_context.shape[-2]) # [traj_size, timesteps, x_dim]
            inputs = np.concatenate([t_context, _ctx_helper_input_initial_cond], axis=-1) # [traj_size, timesteps, x_dim + 1]

            # Encode target and context (context needs to be encoded to calculate kl term)
            mu_context, sigma_context = self.xy_to_z_dist(inputs, x_context, ctx_mask_with_new_traj_obs)
            mu_tgt, sigma_tgt = self.xy_to_z_dist(inputs, x_context, ctx_mask_with_new_traj_target_mask)
            # Sample from encoded distribution using reparameterization trick
            # Get parameters of output distribution
            z_sample = distributions.MultivariateNormalDiag(loc=mu_tgt, scale_diag=sigma_tgt).sample(sample_size, seed=sample_rng)
            z_sample = rearrange(z_sample, "sample_size traj_size z_dim -> traj_size sample_size z_dim")
            y_pred_mu, y_pred_sigma = self.xz_to_y(inputs, z_sample) # [traj_size, sample_size, timesteps, y_dim]
            return y_pred_mu, y_pred_sigma, mu_context, sigma_context, mu_tgt, sigma_tgt 
        else:
            t_context = t_context[..., None] # [traj_size, timesteps, 1]
            _ctx_helper_input_initial_cond = np.repeat(x_context[:, :1, :], axis=-2, repeats=t_context.shape[-2]) # [traj_size, timesteps, x_dim]
            ctx_inputs = np.concatenate([t_context, _ctx_helper_input_initial_cond], axis=-1) # [traj_size, timesteps, x_dim + 1]
            # NOTE: how does this corresponds to t_context
            _tgt_helper_input_initial_cond = np.repeat(x_target[:, :1, :], axis=-2, repeats=t_target.shape[-1]) # [traj_size, timesteps, x_dim]
            tgt_inputs = np.concatenate([t_target[..., None], _tgt_helper_input_initial_cond], axis=-1)
            # At testing time, encode only context
            mu_context, sigma_context = self.xy_to_z_dist(ctx_inputs, x_context, ctx_mask_with_new_traj_obs)
            # Sample from distribution based on context
            q_context = distributions.MultivariateNormalDiag(mu_context, sigma_context)
            z_sample = q_context.sample(sample_size, seed=sample_rng)
            z_sample = rearrange(
                z_sample,
                "sample_size traj_size z_dim -> traj_size sample_size z_dim",
            )
            # Predict target points based on context
            y_pred_mu, y_pred_sigma = self.xz_to_y(tgt_inputs, z_sample)

            return y_pred_mu, y_pred_sigma


class NeuralODEProcess(Module, __AbstractSingleTimeSeriesInputModel):
    x_dim: int
    r_dim: int
    encoder_h_dim: int
    ode_layer_h_dim: int
    decoder_h_dim: int
    latent_d_dim: int
    latent_l_dim: int
    known_initial_condition: bool = False
    tx_to_r_act_fn: str = "relu"
    r_to_z0_act_fn: str = "relu"
    r_to_zd_act_fn: str = "relu"
    z_to_x_act_fn: str = "relu"
    initial_condition_std: Optional[float] = 1e-3
    autonomous_ode: bool = False
    heteroscedastic_noise: bool = True
    fixed_likelihood_noise: bool = False
    decoder_std_lower_bound: Optional[float] = 1E-4
    decoder_std: Optional[float] = 1E-3
    ode_max_step: int = 10000
    ode_step_size_controller_rtol: float = 1e-7
    ode_step_size_controller_atol: float = 1e-9
    z0_lower_std: float = 0.1
    zd_lower_std: float = 0.1
    """
    An implementation of the model structure mentioned in Appendix F of the original paper

    in high level, the Neral ODE Process (NODEP in short) encoder the time series into two task representations
        1. one used to represent the initial states distribution,
        2. the other one is ingeneral used to represent the global control of the ODE's distribution, this part is pragmatically 
            implmented through an augmented state of the ode
            
    Implementation wise:
        dealing with flax based ode part initialization: https://github.com/google/flax/discussions/2891

    Parameters
    ----------
    x_dim: dimensionality of the input states
    r_dim: dimensionality of the encoded representation vector
    encoder_h_dim: dimensioanlity of the unit number of hidden layers in encoder
    ode_layer_h_dim: dimensioanlity of the unit number of hidden layers in ode blocks
    decoder_h_dim: dimensioanlity of the unit number of hidden layers in decoder
    latent_d_dim: the dimensiaonlity of latent variable D representing the global control
    latent_l_dim: the dimensioanlity of the latent state variable L 
    tx_to_r_act_fn: the activation function used in the encoder to encode (t, x) to r, this will affect the differentiability 
        of the model w.r.t x
    r_to_z0_act_fn: the activation function used in the encoder to encode r to z0, this will affect the differentiability
    z_to_x_act_fn: the activation function used in the decoder to decode z to x, this will affect the differentiability
    autonomous_ode: if it is autonomouse ode, the r.h.s of ode is not dependent on time t
        dx/dt = f(x), if it is False then representing dx/dt = f(x, t), note that in Eq. 3 and Eq. 4 of the nodep paper, 
        they mentioned to use time t in both dynamics as well as the decoder, but in their code (AbstractODEDecoder), they offer 
        an option to make it autonomous, we keep this functionality but set it to False to align with the paper

    ode_max_step: the maximum number of steps to take in the ode solver
    ode_step_size_controller_rtol: the relative tolerance of the ode solver, 1e-7 is the default value of the odeint solver
    ode_step_size_controller_atol: the absolute tolerance of the ode solver, 1e-9 is the default value of the odeint solver
    known_initial_condition: whether the initial condition is known, if it is known, 
        for simpicitly of training, it will still map to a distribution with the same dimensionality of the augmented state dim, 
        but provide the addition option of specifying a fixed (small) standard deviation to make sure the distribution is almost 
        concentrated at a same point. The initial condition will be genearted via MuSigma encoder directly through the initial x0 
        (instead of r) initial_condition_std: the standard deviation of the initial condition distribution in the augmented state, 
        NOTE: this is not exists in the original NODEP model
    for the decoder, referes the description of the decoder in Appendix F of the original paper
    fixed_likelihood_noise: whether the likelihood noise is fixed, if it is fixed, the std of the likelihood noise is fixed
        
    """
    def setup(self):
        self._tx_to_r = Encoder(
            self.encoder_h_dim, self.r_dim, act_fn=self.tx_to_r_act_fn
        )
        if not self.known_initial_condition:
            self._r_to_z0_mu_sigma = MuSigmaEncoder(
                self.r_dim,
                self.latent_l_dim,
                std_lower_bound=self.z0_lower_std,
                act_fn=self.r_to_z0_act_fn,
            )
        else:
            assert self.initial_condition_std is not None
            self._x0_to_z0_mu_sigma = MuSigmaFixedLikelihoodEncoder(
                self.x_dim, self.latent_l_dim, std=self.initial_condition_std
            )
        self._r_to_zd_mu_sigma = MuSigmaEncoder(
            self.r_dim,
            self.latent_d_dim,
            std_lower_bound=self.zd_lower_std,
            act_fn=self.r_to_zd_act_fn,
        )
        self.ode_layers = Sequential(
            [
                Dense(self.ode_layer_h_dim),
                tanh,
                Dense(self.ode_layer_h_dim),
                tanh,
                Dense(self.latent_l_dim),
            ]
        )  # refer app F
        if self.heteroscedastic_noise and not self.fixed_likelihood_noise:
            self._z_to_x = HeteroscedasticNODEPDecoder(
                self.decoder_h_dim,
                self.x_dim,
                std_lower_bound=self.decoder_std_lower_bound,
                autonomous_ode=self.autonomous_ode,
                act_fn=self.z_to_x_act_fn,
            )
        elif self.fixed_likelihood_noise:
            self._z_to_x = FixedLikelihoodNODEPDecoder(
                self.decoder_h_dim,
                self.x_dim,
                std=self.decoder_std,
                autonomous_ode=self.autonomous_ode,
                act_fn=self.z_to_x_act_fn,
            )
        else:
            raise NotImplementedError("Decoder Type Not Understood!")
        self.ode_f = lambda z, global_control, t: self.apply(
            self.variables, z, global_control, t, method=self.ode
        )

    def aggregate(self, r_i, mask=None):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : [num_points, r_dim]
        mask: [num_points], boolean mask specifying which element is needed to
            calculated the aggregation, this is mainly utilized to keep NN input same shape
            to be JIT friednly as then no need for recompilation
        """
        if mask == None:
            return np.mean(r_i, axis=-2)
        else:
            return np.sum(r_i * mask[..., None], axis=-2) / np.sum(
                mask, axis=-1, keepdims=True
            )

    def tx_to_z_dist(self, t, x, mask=None):
        """
        Maps (t, x) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        t : Shape (timesteps)
        x : Shape (num_points, x_dim)
        mask: (num_points)
        """
        # Encode each point into a representation r_i
        # note that we assume the data are observed in an isotopic fashion: each state are observed simultaneously
        r_i = self._tx_to_r(
            np.concatenate([np.expand_dims(t, -1), x], -1)
        )  # [timesteps, r_dim]
        # Aggregate representations r_i into a single representation r
        # we delibrately make use the property of mean scalarization to take mask into account
        r = self.aggregate(r_i, mask)  # [r_dim]
        if not self.known_initial_condition:
            # Return parameters of distribution
            return self._r_to_z0_mu_sigma(r), self._r_to_zd_mu_sigma(r)
        else:
            return self._x0_to_z0_mu_sigma(x[0, ...]), self._r_to_zd_mu_sigma(r)

    def ode(self, z_state, global_control, t):
        """
        calculate the dynamics of ode: f(x) or f(x, t)
        z_state: (..., state_dim)
        t: float
        global_control: (..., state_dim)
        """
        if not self.autonomous_ode:
            if len(z_state.shape) == 1:
                expand_t = np.reshape(t, (-1,))
            else:
                expand_t = np.full(z_state.shape[:-1] + (1,), t)
            # expand_t = np.broadcast_to(np.atleast_2d(t), shape=(z_state.shape[0], 1))
            ode_input = np.concatenate([z_state, global_control, expand_t], axis=-1)
        else:  # autonomous ode
            ode_input = np.concatenate([z_state, global_control], axis=-1)
        return self.ode_layers(ode_input)

    def solve_ode(self, y0, save_ts, global_control, _t0, _t1, solver):
        """
        y0: [traj_size, sample_size, state_dim]
        """
        def ode_f(t, z, args):
            _control = args
            return self.ode_f(z, _control, t)

        ode_step_size_controller_rtol = self.ode_step_size_controller_rtol
        ode_step_size_controller_atol = self.ode_step_size_controller_atol
        res = diffeqsolve(
            ODETerm(ode_f),
            t0=_t0,
            t1=_t1,
            dt0=None,
            solver=getattr(diffrax, solver)(),
            stepsize_controller=PIDController(
                rtol=ode_step_size_controller_rtol,
                atol=ode_step_size_controller_atol,
            ),
            args=global_control,
            max_steps=self.ode_max_step,
            y0=y0,
            throw=False,
            saveat=diffrax.SaveAt(ts=save_ts),
        )
        return res.ys, res.stats["num_steps"]

    def __call__(
        self,
        t_context,
        x_context,
        t_target,
        sample_rng=None,
        context_mask=None,
        target_mask=None,
        sample_size: int = None,
        x_target=None,
        training=None,
        solver="Dopri5",
        t0=None,
        t1=None,
        **kwargs
    ):
        """

        The solver largely follows the setting of
        https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/odeint.py#L33
        which is provided in the original neural ode paper

        We assume
            - t_context is a subset of t_target
            - t_context and t_target is ordered, note this is a MUST DO

        Parameters
        ----------
        t_context : [timesteps]
        x_context : [timesteps, x_dim]. Note that x_context is a subset of x_target.
        t_target : [timesteps']
        x_target : Optional [timesteps', x_dim]. Only used during training.

        training: future drop out usage

        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        if training:
            # Encode target and context (context needs to be encoded to calculate kl term)
            (mu_z0_ctx, sigma_z0_ctx), (mu_global_ctx, sigma_global_ctx) = (
                self.tx_to_z_dist(t_context, x_context, context_mask)
            )  # (batch_size, z_dim), (batch_size, z_latent_dim)
            (mu_z0_tgt, sigma_z0_tgt), (mu_global_tgt, sigma_global_tgt) = (
                self.tx_to_z_dist(t_target, x_target, target_mask)
            )  # (batch_size, z_dim), (batch_size, z_latent_dim)
            # Sample from encoded distribution using reparameterization trick

            # note this sample is needed, since it is not sample average approximation
            _, z0_rng, z_global_rng = random.split(sample_rng, 3)

            # reparam sampling the initial and control: note that MultivariateNormalDiag is FULLY_REPARAMETERIZED hence no need to manually implement reparam sampling
            sampled_z0 = distributions.MultivariateNormalDiag(
                mu_z0_tgt, sigma_z0_tgt
            ).sample(sample_size, seed=z0_rng)
            sampled_global_control = distributions.MultivariateNormalDiag(
                mu_global_tgt, sigma_global_tgt
            ).sample(sample_size, seed=z_global_rng)

            # hack to init ode layers params due to jax lazy flax initialization
            _ = self.ode(sampled_z0[:1], sampled_global_control[:1], np.atleast_2d(t0))
            z_aug, ode_step = self.solve_ode(
                sampled_z0, t_target, sampled_global_control, t0, t1, solver
            )  # (timestep, sample_size, state_dim)
            z_t_sample = rearrange(
                z_aug,
                "timestep sample_size state_dim -> sample_size timestep state_dim",
            )
            y_pred_mu, y_pred_sigma = self._z_to_x(
                t_target, z_t_sample, sampled_global_control
            )  # [sample_size, timestep, y_dim]

            return (
                y_pred_mu,
                y_pred_sigma,
                mu_z0_ctx,
                sigma_z0_ctx,
                mu_z0_tgt,
                sigma_z0_tgt,
                mu_global_ctx,
                sigma_global_ctx,
                mu_global_tgt,
                sigma_global_tgt,
                ode_step,
            )
        else:
            # At testing time, encode only context
            (mu_z0_ctx, sigma_z0_ctx), (mu_global_ctx, sigma_global_ctx) = (
                self.tx_to_z_dist(t_context, x_context, context_mask)
            )  # (z_dim), (z_latent_dim)
            # Sample from distribution based on context
            q_z0_context = distributions.MultivariateNormalDiag(mu_z0_ctx, sigma_z0_ctx)
            q_global_context = distributions.MultivariateNormalDiag(
                mu_global_ctx, sigma_global_ctx
            )
            _, z0_rng, z_global_rng = random.split(sample_rng, 3)
            sampled_z0 = q_z0_context.sample(sample_size, seed=z0_rng)
            sampled_global_control = q_global_context.sample(sample_size, z_global_rng)

            # hack to init ode layers params due to jax lazy flax initialization
            _ = self.ode(sampled_z0[:1], sampled_global_control[:1], np.atleast_2d(t0))

            z_aug, ode_step = self.solve_ode(
                sampled_z0, t_target, sampled_global_control, t0, t1, solver
            )  # (timestep, sample_size, state_dim)
            z_t_sample = rearrange(
                z_aug,
                "timestep sample_size state_dim -> sample_size timestep state_dim",
            )
            y_pred_mu, y_pred_sigma = self._z_to_x(
                t_target, z_t_sample, sampled_global_control
            )  # [sample_size, timestep, y_dim]
            # Predict target points based on context
            return y_pred_mu, y_pred_sigma


class NeuralODEProcessAcceptMultiTimeSeriesData(NeuralODEProcess, __AbstractMultiTimeSeriesInputModel):
    """
    An extended NeuralODEProcee model, the only difference of this model with the Neural ODE Process 
    is that it accepts multi trajectories data from the same dynamic system, so it utilize the same 
    training data flow as SANODEP, which can be serve as a good comparison of NODEP vs SANODEP

    Note that follow NODEP, the model still treat each trajectory independently as in NeuralODEProcess 
    
    when training, it will return the output of all the trajectories
    when testing, it will only return the output of the first trajectories since we assume we are only interested in the 
    first trajectory prediction by observing all the rest trajectories 
    """    
    def aggregate(self, r_i, mask = None):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : [traj_size, num_points, r_dim]
        mask: [traj_size, num_points], boolean mask specifying which element is needed to 
            calculated the aggregation, this is mainly utilized to keep NN input same shape 
            to be JIT friednly as then no need for recompilation
        """
        if mask == None:
            traj_wise_aggregation = np.mean(r_i, axis=-2) # [traj_size, r_dim]
            return traj_wise_aggregation
        else:
            # note that since a trajectory can all have 0 value, hence when average we assume the 
            # denominator is assumed to be at least 1, this will not affect the traj_wise_aggregation 
            # value (0) if the whole trajectory is 0
            traj_wise_aggregation = np.sum(r_i * mask[..., None], axis=-2) / np.maximum(np.sum(mask, axis=-1, keepdims=True), 1) # [traj_size, num_points]
            return traj_wise_aggregation

    def tx_to_z_dist(self, t, x, mask=None):
        """
        Maps (t, x) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        t : Shape (traj_size, timesteps)
            The time values for each trajectory.
        x : Shape (traj_size, num_points, x_dim)
            The input data points for each trajectory.
        mask: (traj_size, num_points)
            The mask indicating which points are valid for each trajectory.

        Returns
        -------
        Tuple
            A tuple containing the mu and sigma parameters of the latent variables z.
            - For known initial condition:
                - mu_sigma_z0 : Tuple
                    A tuple containing the mu and sigma parameters of the initial latent variable z0.
                - mu_sigma_zd : Tuple
                    A tuple containing the mu and sigma parameters of the latent variable zd.
            - For unknown initial condition:
                - mu_sigma_z0 : Tuple
                    A tuple containing the mu and sigma parameters of the latent variable z0, derived from x0.
                - mu_sigma_zd : Tuple
                    A tuple containing the mu and sigma parameters of the latent variable zd.
        """
        # Encode each point into a representation r_i
        r_i = self._tx_to_r(np.concatenate([np.expand_dims(t, -1), x], -1)) # [traj, timesteps, r_dim]
        # Aggregate representations r_i into a single representation r
        # we delibrately make use the property of mean scalarization to take mask into account
        r = self.aggregate(r_i, mask)  # [traj_size, r_dim]
        return self._r_to_z0_mu_sigma(r), self._r_to_zd_mu_sigma(r)
        
    def __call__(self, t_context, x_context, t_target, sample_rng = None,
                 context_mask = None, target_mask = None, sample_size: int = None, 
                 x_target=None, training=None, solver='Dopri5', t0 = None, t1 = None, **kwargs):
        """

        The solver largely follows the setting of 
        https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/odeint.py#L33
        which is provided in the original neural ode paper

        We assume 
            - t_context is a subset of t_target
            - t_context and t_target is ordered, note this is a MUST DO

        Parameters
        ----------
        t_context : [traj_size, timesteps] 
        x_context : [traj_size, timesteps, x_dim]. Note that x_context is a subset of x_target.
        t_target : [traj_size, timesteps']
        x_target : Optional [traj_size, timesteps', x_dim]. Only used during training.


        training: future drop out usage

        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        if training:
            # Encode target and context (context needs to be encoded to calculate kl term)
            (mu_z0_ctx, sigma_z0_ctx), (mu_global_ctx, sigma_global_ctx) = self.tx_to_z_dist(t_context, x_context, context_mask) # (batch_size, z_dim), (batch_size, z_latent_dim)
            (mu_z0_tgt, sigma_z0_tgt), (mu_global_tgt, sigma_global_tgt) = self.tx_to_z_dist(t_target, x_target, target_mask) # (batch_size, z_dim), (batch_size, z_latent_dim)
            # Sample from encoded distribution using reparameterization trick

            # note this sample is needed, since it is not sample average approximation
            _, z0_rng, z_global_rng = random.split(sample_rng, 3)
            # reparam sampling the initial and control 

            # reparam sampling
            sampled_z0 = distributions.MultivariateNormalDiag(loc=mu_z0_tgt, scale_diag=sigma_z0_tgt).sample(sample_size, seed=z0_rng)
            sampled_z0 = rearrange(sampled_z0, "sample_size traj_size z_dim -> traj_size sample_size z_dim")

            # sampled_global_control = np.expand_dims(mu_global_tgt, axis=1) + \
            #     np.expand_dims(sigma_global_tgt, axis=1) * random.normal(z_global_rng, shape=(traj_size, sample_size, self.latent_l_dim))
            sampled_global_control = distributions.MultivariateNormalDiag(loc=mu_global_tgt, scale_diag=sigma_global_tgt).sample(sample_size, seed=z_global_rng)
            sampled_global_control = rearrange(sampled_global_control, "sample_size traj_size z_dim -> traj_size sample_size z_dim")
            # hack to init ode layers params due to jax lazy flax initialization
            _ = self.ode(sampled_z0[:1], sampled_global_control[:1], np.atleast_3d(t0))

            # note that, here a tricky things have been made which is to only use the 1st one within a trajectory batch
            z_aug, ode_step = self.solve_ode(sampled_z0, t_target[0], sampled_global_control, t0, t1, solver) # (timestep, sample_size, state_dim)
            z_t_sample = rearrange(z_aug, "timestep traj_size sample_size state_dim -> traj_size sample_size timestep state_dim")
            y_pred_mu, y_pred_sigma = self._z_to_x(t_target[0], z_t_sample, sampled_global_control) # [sample_size, timestep, y_dim]

            return y_pred_mu, y_pred_sigma, mu_z0_ctx, sigma_z0_ctx, mu_z0_tgt, sigma_z0_tgt, mu_global_ctx, sigma_global_ctx, mu_global_tgt, sigma_global_tgt, ode_step
        else:
            # At testing time, encode only context
            (mu_z0_ctx, sigma_z0_ctx), (mu_global_ctx, sigma_global_ctx) = self.tx_to_z_dist(t_context, x_context, context_mask) # (z_dim), (z_latent_dim)
            # Sample from distribution based on context
            q_z0_context = distributions.MultivariateNormalDiag(mu_z0_ctx, sigma_z0_ctx)
            q_global_context = distributions.MultivariateNormalDiag(mu_global_ctx, sigma_global_ctx)
            _, z0_rng, z_global_rng = random.split(sample_rng, 3)
            # traj_size = t_context.shape[0]
            sampled_z0 = q_z0_context.sample(sample_shape=(sample_size), seed=z0_rng) # [sample_size, traj_size, z_dim]
            sampled_z0 = rearrange(sampled_z0, "sample_size traj_size z_dim -> traj_size sample_size z_dim")
            sampled_global_control = q_global_context.sample(sample_shape=(sample_size), seed=z_global_rng) # [sample_size, traj_size, z_dim]
            sampled_global_control = rearrange(sampled_global_control, "sample_size traj_size z_dim -> traj_size sample_size z_dim")

            _ = self.ode(sampled_z0[:1], sampled_global_control[:1], np.atleast_3d(t0))            
            z_aug, ode_step = self.solve_ode(sampled_z0, t_target[0], sampled_global_control, t0, t1, solver) # (timestep, sample_size, state_dim)
            z_t_sample = rearrange(z_aug, "timestep traj_size sample_size state_dim -> traj_size sample_size timestep state_dim")
            y_pred_mu, y_pred_sigma = self._z_to_x(t_target[0], z_t_sample, sampled_global_control) # [sample_size, timestep, y_dim]
            # Predict target points based on context
            return y_pred_mu, y_pred_sigma    


class __AbstractSystemAwareNueralODEProcess(
    NeuralODEProcess, __AbstractMultiTimeSeriesInputModel
):
    """
    A neural process model that accepts multi timeseries starting at different initial conditions from the same dynamic system
    This mdoel classs and it inheritances are mainly used to learn dynamic systems from multiple trajectories
    """

    @classmethod
    @abstractmethod
    def aggregate(self, r_i, mask=None):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : [traj_size, num_points, r_dim]
        mask: [traj_size, num_points], boolean mask specifying which element is needed to
            calculated the aggregation, this is mainly utilized to keep NN input same shape
            to be JIT friednly as then no need for recompilation
        """

    @classmethod
    @abstractmethod
    def __call__(
        self,
        t_context,
        x_context,
        t_target,
        sample_rng=None,
        context_mask=None,
        target_mask=None,
        sample_size: int = None,
        x_target=None,
        training=None,
        solver="Dopri5",
        t0=None,
        t1=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        t_context : [traj_size, timesteps]
        x_context : [traj_size, timesteps, x_dim]. Note that x_context is a subset of x_target.
        t_target : [traj_size, timesteps']
        x_target : Optional [traj_size, timesteps', x_dim]. Only used during training.
        """

    @classmethod
    @abstractmethod
    def aggregate_r_system(self):
        """
        extract system feature from aggregated vector r_i
        """


class SANODEP(__AbstractSystemAwareNueralODEProcess):
    """
    A workable meta model to meta learn the across different dynamic systems

    This is very much like MultiTrajectoryAwareNeuralODEProcessDeepSetsAggregation
    but the model is defined as dx/dt = f(x, t, dsys), where dsys is system wise 
    aggregation, so there is no "trajectory wise dynamics" anymore

    Importantly, be aware of the model __call__ function works in a specific mechanism to 
    meta learn from multiple trajectories of a same dynamic system 
    """

    t_embedding: Optional[str] = None
    t_embedding_dim: Optional[int] = 16
    # used in t_embedding, similar as the maximum length used in positional encoding, not used when t_embedding is None
    maximum_timescale: float = np.inf  
    d_sys_lower_std: float = 0.1
    tx0_to_r_global_act_fn: str = "relu"
    r_to_d_sys_mu_sigma: str = "relu"

    def setup(self):
        super().setup()
        self._tx0x_to_r_global = Encoder(
            self.encoder_h_dim, self.r_dim, act_fn=self.tx0_to_r_global_act_fn
        )
        self._r_to_d_sys_mu_sigma = MuSigmaEncoder(
            self.r_dim,
            self.latent_d_dim,
            std_lower_bound=self.d_sys_lower_std,
            act_fn=self.r_to_d_sys_mu_sigma,
        )

        if self.t_embedding == "SEFT":
            from .embedding import SEFTTimeEncoding

            self.t_emb = SEFTTimeEncoding(self.t_embedding_dim)
        elif self.t_embedding == "mTAND":
            raise NotImplementedError
        elif self.t_embedding == "ROPE":
            raise NotImplementedError
        else:
            self.t_emb = lambda x, **kwargs: x

    def aggregate_r_system(self, r_aug_i, mask=None):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation. Importantly, be aware of the additional traj_size axis in mask, 
        this is because the first axis of mask serves as a Monte Carlo sampling dim (a.k.a., batch dim)
        so actually the operation between r_i and mask is
        for _mask in mask:
            r_i * _mask[..., None]
        here it has been implemented in a batchable way

        Parameters
        ----------
        r_i : [traj_size, num_points, r_dim]
        mask: [traj_size, traj_size, num_points], boolean mask specifying which element is needed to
            calculated the aggregation, this is mainly utilized to keep NN input same shape
            to be JIT friednly as then no need for recompilation
        return [traj_size, r_dim]
        """
        # here we have to do this rearrange because we need to flatten to the summation
        flatten_r_i = rearrange(
            r_aug_i, "traj_size num_points r_dim -> (traj_size num_points) r_dim"
        )
        flatten_mask = rearrange(
            mask,
            "traj_size1 traj_size2 num_points -> traj_size1 (traj_size2 num_points)",
        )  # [traj_size, traj_size * num_points]
        # take the mean of the last axis across thoses, the usage of this looks bit weird implementation
        # is only to ensure that the average is operated across those masked True elements
        r_system = np.sum(flatten_r_i * flatten_mask[..., None], axis=-2) / np.maximum(
            np.sum(flatten_mask, axis=-1, keepdims=True), 1
        )
        return r_system

    def aggregate_r0(self, r_i, mask=None):
        """
        Aggregates representations for every (x_i, y_i) pair in a single trajectory into a single
        representation.

        Parameters
        ----------
        r_i : [traj_size, num_points, r_dim]
        mask: [traj_size, num_points], boolean mask specifying which element is needed to
            calculated the aggregation, this is mainly utilized to keep NN input same shape
            to be JIT friednly as then no need for recompilation
        """
        if mask == None:
            traj_wise_aggregation = np.mean(r_i, axis=-2)
            return traj_wise_aggregation  # [traj_size, r_dim]
        else:
            # note that since a trajectory can all have 0 value (means the full trajectory is unobserved),
            # hence when average we assume the denominator is at least 1, this will not affect the traj_wise_aggregation
            # value (0) if the whole trajectory is 0
            traj_wise_aggregation = np.sum(r_i * mask[..., None], axis=-2) / np.maximum(
                np.sum(mask, axis=-1, keepdims=True), 1
            )  # [traj_size, num_points]
            return (
                traj_wise_aggregation * np.any(mask, axis=-1)[..., None]
            )  # screen out not used trajectories [traj_size]
        
    def aggregate(self, r_i, mask=None):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : [traj_size, num_points, r_dim]
        mask: [traj_size, num_points], boolean mask specifying which element is needed to
            calculated the aggregation, this is mainly utilized to keep NN input same shape
            to be JIT friednly as then no need for recompilation
        """
        if mask == None:
            traj_wise_aggregation = np.mean(r_i, axis=-2)  # [traj_size, r_dim]
            return traj_wise_aggregation
        else:
            # note that since a trajectory can all have 0 value, hence when average we assume the
            # denominator is assumed to be at least 1, this will not affect the traj_wise_aggregation
            # value (0) if the whole trajectory is 0
            traj_wise_aggregation = np.sum(r_i * mask[..., None], axis=-2) / np.maximum(
                np.sum(mask, axis=-1, keepdims=True), 1
            )  # [traj_size, num_points]
            return traj_wise_aggregation

    def tx_to_z0_dist(self, t, x, mask=None):
        """
        Maps (t, x) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        t : Shape (traj_size, timesteps)
        x : Shape (traj_size, num_points, x_dim)
        mask: (traj_size, num_points)
        """
        tx = np.concatenate([t[..., None], x], -1)
        r_i = self._tx_to_r(tx)  # [traj, timesteps, r_dim]
        r0_all_traj = self.aggregate_r0(r_i, mask)  # [traj, r_dim]
        return self._r_to_z0_mu_sigma(r0_all_traj)

    def tx_to_system(self, t, x, init_cond, mask=None):
        """
        for each of the predicting trajecory, extract the corresponding dynamic system feature vector
        t: [traj_size, num_points]
        x: [traj_size, num_points, state_dim]
        init_cond: [traj_size, z_dim]
        mask: [traj_size, traj_size, num_points]

        for each of the trajectory (to predict), there are sample_size dynamic system features
        return [traj_size, z_dim], [traj_size, z_dim]
        """
        # _sample_size = init_cond.shape[1]
        t_emb = self.t_emb(
            t[..., None], max_period=self.maximum_timescale
        )  # [traj_size, timesteps, 1]
        init_cond = np.broadcast_to(
            np.expand_dims(init_cond, -2), (x.shape[:-1] + (init_cond.shape[-1],))
        )  # [traj_size, num_points, z_dim]
        tx_aug_infered_init_cond = np.concatenate(
            [t_emb, init_cond, x], -1
        )  # [traj_size, timesteps, time_dim+latent_z0+state_dim]
        aug_init_ri = self._tx0x_to_r_global(
            tx_aug_infered_init_cond
        )  # [traj, timesteps, r_dim]
        rd_system = self.aggregate_r_system(
            aug_init_ri, mask=mask
        )  # [traj, timesteps, r_dim], [traj_size, traj_size, num_points] -> [traj_size, r_dim]
        return self._r_to_d_sys_mu_sigma(rd_system)

    def __call__(
        self,
        t_context,
        x_context,
        t_target,
        sample_rng,
        target_initial_cond_mask,
        ctx_mask_with_new_traj_obs,
        ctx_mask_with_new_traj_target_mask,
        sample_size: int,
        x_target,
        training,
        t0=None,
        t1=None,
        solver="Dopri5",
        **kwargs
    ):
        """
        The solver largely follows the setting of
        https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/odeint.py#L33
        which is provided in the original neural ode paper

        We assume
            - t_context is a subset of t_target
            - t_context and t_target is ordered, note this is a MUST DO
            - when training, all the target trajectory has the same time scheduling, this is a MUST DO and makes the ode solve parallelable

        Parameters
        ----------
        t_context : [traj_size, timesteps]
        x_context : [traj_size, timesteps, x_dim]. Note that x_context is a subset of x_target.
        t_target : [traj_size, timesteps']
        x_target : Optional [traj_size, timesteps', x_dim]. Only used during training.
        context_mask: [traj_size, num_points], boolean mask specifying which element is needed to
        target_mask: [traj_size, num_points], boolean mask specifying which element is needed to
        ctx_mask_with_new_traj_obs: [traj_size, traj_size, num_points], this could represent problem setting 1
            and problem setting 2
        ctx_mask_with_new_traj_target_mask: [traj_size, traj_size, num_points]

        training: future drop out usage

        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        if training:
            # calculate the distribution of the latent initial condition
            mu_z0_tgt, sigma_z0_tgt = self.tx_to_z0_dist(
                t_target, x_target, target_initial_cond_mask
            )  # (traj_size, z_dim), (traj_size, z_latent_dim)

            _, z0_rng, z_system_rng = random.split(sample_rng, 3)
            # sample the initial condition for latent ODE solving
            sampled_z0_tgt = distributions.MultivariateNormalDiag(
                mu_z0_tgt, sigma_z0_tgt
            ).sample(
                sample_shape=(sample_size), seed=z0_rng
            )  # [init_sample_size, traj_size, z_dim]
            sampled_z0_tgt = rearrange(
                sampled_z0_tgt,
                "init_sample_size traj_size z_dim -> traj_size init_sample_size z_dim",
            )

            known_initial_cond = x_context[
                ..., 0, :
            ]  # [traj_size, state] get the first timestep, this need to assume that all the data known the initial condition!

            # extract the amortized vi's mean and std based on context data 
            mu_sys_dynamic_ctx, sigma_sys_dynamic_ctx = self.tx_to_system(
                t_context,
                x_context,
                known_initial_cond,
                ctx_mask_with_new_traj_obs,
            )  # [traj_size, z_dim]

            # extract the amortized vi's mean and std based on target data
            mu_sys_dynamic_tgt, sigma_sys_dynamic_tgt = self.tx_to_system(
                t_target,
                x_target,
                known_initial_cond,
                ctx_mask_with_new_traj_target_mask,
            )  # [traj_size, z_dim]

            # refer loss function derive in the paper
            sampled_system_dynamics = distributions.MultivariateNormalDiag(
                mu_sys_dynamic_tgt, sigma_sys_dynamic_tgt
            ).sample(
                sample_shape=(sample_size,), seed=z_system_rng
            )  # [sample_size, traj_size, z_dim]
            sampled_system_dynamics = rearrange(
                sampled_system_dynamics,
                "sample_size traj_size z_dim -> traj_size sample_size z_dim",
            )
            # hack to please Flax lazy initialization: here we use this to initialize self.ode's params
            _ = self.ode(
                sampled_z0_tgt[:1], sampled_system_dynamics[:1], np.atleast_3d(t0)
            )

            # hack: we assume all the target trajectory has the same time scheduling such that we can simply make use of the first trajectory's schedule
            z_solution, ode_step = self.solve_ode(
                sampled_z0_tgt, t_target[0], sampled_system_dynamics, t0, t1, solver
            )  # (timestep, traj_size, sample_size, state_dim)
            z_t_sample = rearrange(
                z_solution,
                "timestep traj_size sample_size state_dim -> traj_size sample_size timestep state_dim",
            )

            # hack: we assume all the target trajectory has the same time scheduling such that we can simply make use of the first trajectory's schedule
            y_pred_mu, y_pred_sigma = self._z_to_x(
                t_target[0], z_t_sample, sampled_system_dynamics
            )  # [traj_size, sample_size, timestep, y_dim]
            return (
                y_pred_mu,
                y_pred_sigma,
                mu_z0_tgt,
                sigma_z0_tgt,
                mu_sys_dynamic_ctx,
                sigma_sys_dynamic_ctx,
                mu_sys_dynamic_tgt,
                sigma_sys_dynamic_tgt,
                ode_step,
            )
        
        else:
            known_initial_cond = x_context[..., 0, :]
            mu_z0_tgt, sigma_z0_tgt = self.tx_to_z0_dist(
                t_context, x_context, target_initial_cond_mask
            )
            q_z0_tgt = distributions.MultivariateNormalDiag(mu_z0_tgt, sigma_z0_tgt)

            _, z0_rng, z_system_rng = random.split(sample_rng, 3)
            sampled_z0_tgt = q_z0_tgt.sample(
                sample_shape=(sample_size), seed=z0_rng
            )  # [sample_size, traj_size, z_dim]
            sampled_z0_tgt = rearrange(
                sampled_z0_tgt,
                "init_sample_size traj_size z_dim -> traj_size init_sample_size z_dim",
            )

            mu_sys_dynamic_ctx, sigma_sys_dynamic_ctx = self.tx_to_system(
                t_context,
                x_context,
                known_initial_cond,
                ctx_mask_with_new_traj_obs,
            )

            sampled_system_dynamics = distributions.MultivariateNormalDiag(
                mu_sys_dynamic_ctx, sigma_sys_dynamic_ctx
            ).sample(
                sample_shape=(sample_size,), seed=z_system_rng
            )  # [traj_size, sample_size, z_dim]
            sampled_system_dynamics = rearrange(
                sampled_system_dynamics,
                "sample_size traj_size z_dim -> traj_size sample_size z_dim",
            )
            # hack to please Flax lazy initialization: here we use this to initialize self.ode's params
            _ = self.ode(
                sampled_z0_tgt[:1], sampled_system_dynamics[:1], np.atleast_3d(t0)
            )
            # hack: we assume all the target trajectory has the same time scheduling such that we can simply make use of the first trajectory's schedule
            z_solution, ode_step = self.solve_ode(
                sampled_z0_tgt, t_target[0], sampled_system_dynamics, t0, t1, solver
            )  # (timestep, sample_size, state_dim)
            z_t_sample = rearrange(
                z_solution,
                "timestep traj_size sample_size state_dim -> traj_size sample_size timestep state_dim",
            )
            # hack: we assume all the target trajectory has the same time scheduling such that we can simply make use of the first trajectory's schedule
            y_pred_mu, y_pred_sigma = self._z_to_x(
                t_target[0], z_t_sample, sampled_system_dynamics
            )  # [traj_size, sample_size, timestep, y_dim]
            # Predict target points based on context
            return (
                y_pred_mu,
                y_pred_sigma,
            )  # tricky: we only return the first in all trajectories so we can compare with the original implementation


class SANODEP_Encoder_Without_Initial_Condition(SANODEP):
    def tx_to_system(self, t, x, init_cond, mask=None):
        """
        for each of the predicting trajecory, extract the corresponding dynamic system feature vector
        t: [traj_size, num_points]
        x: [traj_size, num_points, state_dim]
        init_cond: [traj_size, z_dim]
        mask: [traj_size, traj_size, num_points]

        for each of the trajectory (to predict), there are sample_size dynamic system features
        return [traj_size, z_dim], [traj_size, z_dim]
        """
        # _sample_size = init_cond.shape[1]
        t_emb = self.t_emb(
            t[..., None], max_period=self.maximum_timescale
        )  # [traj_size, timesteps, 1]
        tx_aug_infered_init_cond = np.concatenate(
            [t_emb, x], -1
        )  # [traj_size, timesteps, time_dim+latent_z0+state_dim]
        aug_init_ri = self._tx0x_to_r_global(
            tx_aug_infered_init_cond
        )  # [traj, timesteps, r_dim]
        rd_system = self.aggregate_r_system(
            aug_init_ri, mask=mask
        )  # [traj, timesteps, r_dim], [traj_size, traj_size, num_points] -> [traj_size, r_dim]
        return self._r_to_d_sys_mu_sigma(rd_system)


class PI_SANODEP(SANODEP, ABC):
    stability_state_value_replacing: float = 1e6
    time_scaling_coefficient: float = np.nan # the coefficient to scale the time, use nan to force specification
    param_clamping: callable = lambda x: (x, np.zeros(shape=x.shape[:-1])) # the mapping function to map the parameters to the desired space
    mu_param_mapping: object = IdentityTransform()
    sigma_param_mapping: object = IdentityTransform()
    param_est_ff_layers: int = 1
    return_param_est: bool = False
    """
    :params stability_state_value_clamping: it could be possible that the inferred parameter setiing can generate diverging 
         trajectories, this is typically obsered in PI-SANODEP for some dynamical system (e.g., Brusselator), here we replace 
         thoese np.inf in diverging trajectories with this value as a pragmatic work around.
    :params time_scaling_coefficient (float): 
        the coefficient to scale the time. we follow NODEP paper to scale the data set time
        when modeling in SANODEP. However, in PI-SANODEP, since the vector field parameterization is explicit and exactly trying 
        to match the vector field of the dataset (i.e., the true dynamics), we cannot scale time anymore. Hence, we use this 
        parameter to scale the time back to the original time scale as a patch. 
    :param_clampping for some dynamical system, the parameter estimation at the initial training stage can be far from the 
        correct one hence having divergent trajectories, param_clampping is a lambda function to regularize such samples to enforce
        the stability of training process
    :param mu_param_mapping transform the output of estimated parameters to the desired space, by default it is identity transform
    :param_sigma_param_mapping transform the output of estimated parameters to the desired space, by default it is identity transform
        training process when the estimated parameter range have highly different scales across dimensions (e,g., the reactor net case)
    :param_est_ff_layers hidden layers in the encoder, by default it is 1, if not otherwise specified, tx_to_system method is exactly 
        the same as the one in SANODEP
    :param return_param_est: whether to return the estimated parameters, by default it is False
    """
    
    def setup(self):
        """
        Specify the problem dependent decoder here
        """
        self._pi_z_to_x = PI_HeteroscedasticNODEPDecoder(
                self.decoder_h_dim,
                self.x_dim,
                std_lower_bound=self.decoder_std_lower_bound,
                autonomous_ode=self.autonomous_ode,
                act_fn=self.z_to_x_act_fn,
            )
        self._pi_r_to_d_sys_mu_sigma = MuSigmaEncoder(
            self.r_dim,
            self.latent_d_dim,
            std_lower_bound=self.d_sys_lower_std,
            act_fn=self.r_to_d_sys_mu_sigma,
            hidden_layers=self.param_est_ff_layers,
        )
        SANODEP.setup(self)

    def tx_to_system(self, t, x, init_cond, mask=None):
        """
        for each of the predicting trajecory, extract the corresponding dynamic system feature vector
        t: [traj_size, num_points]
        x: [traj_size, num_points, state_dim]
        init_cond: [traj_size, z_dim]
        mask: [traj_size, traj_size, num_points]

        for each of the trajectory (to predict), there are sample_size dynamic system features
        return [traj_size, z_dim], [traj_size, z_dim]
        """
        # _sample_size = init_cond.shape[1]
        t_emb = self.t_emb(
            t[..., None], max_period=self.maximum_timescale
        )  # [traj_size, timesteps, 1]
        init_cond = np.broadcast_to(
            np.expand_dims(init_cond, -2), (x.shape[:-1] + (init_cond.shape[-1],))
        )  # [traj_size, num_points, z_dim]
        tx_aug_infered_init_cond = np.concatenate(
            [t_emb, init_cond, x], -1
        )  # [traj_size, timesteps, time_dim+latent_z0+state_dim]
        aug_init_ri = self._tx0x_to_r_global(
            tx_aug_infered_init_cond
        )  # [traj, timesteps, r_dim]
        rd_system = self.aggregate_r_system(
            aug_init_ri, mask=mask
        )  # [traj, timesteps, r_dim], [traj_size, traj_size, num_points] -> [traj_size, r_dim]
        return self._pi_r_to_d_sys_mu_sigma(rd_system)

    def solve_ode(self, y0, save_ts, global_control, _t0, _t1, solver, replace_exception_with_inf: bool=False):
        def ode_f(t, z, args):
            _control = args
            return self.ode_f(z, _control, t)

        # while True:
        ode_step_size_controller_rtol = self.ode_step_size_controller_rtol
        ode_step_size_controller_atol = self.ode_step_size_controller_atol
        try:
            res = diffeqsolve(
                ODETerm(ode_f),
                t0=_t0,
                t1=_t1,
                dt0=None,
                solver=getattr(diffrax, solver)(),
                stepsize_controller=PIDController(
                    rtol=ode_step_size_controller_rtol,
                    atol=ode_step_size_controller_atol,
                ),
                args=global_control,
                max_steps=self.ode_max_step,
                y0=y0,
                saveat=diffrax.SaveAt(ts=save_ts),
                throw=False
            )
            # replace np.inf in the trajectory with the stability_state_value_replacing
            res_ys = res.ys
        except:
            if replace_exception_with_inf:
                res_ys = np.repeat((np.ones_like(y0) * np.inf)[None, ...], save_ts.shape[0], axis=0)
            else: # through the exception
                raise
        screened_ys = np.where(res_ys == np.inf, self.stability_state_value_replacing, res_ys)
        return screened_ys, res.stats["num_steps"]
        
    def __call__(
        self,
        t_context,
        x_context,
        t_target,
        sample_rng,
        target_initial_cond_mask,
        ctx_mask_with_new_traj_obs,
        ctx_mask_with_new_traj_target_mask,
        sample_size: int,
        x_target,
        training,
        t0=None,
        t1=None,
        solver="Dopri5",
        **kwargs
    ):
        """
        The solver largely follows the setting of
        https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/odeint.py#L33
        which is provided in the original neural ode paper

        We assume
            - t_context is a subset of t_target
            - t_context and t_target is ordered, note this is a MUST DO
            - when training, all the target trajectory has the same time scheduling, this is a MUST DO and makes the ode solve parallelable

        Parameters
        ----------
        t_context : [traj_size, timesteps]
        x_context : [traj_size, timesteps, x_dim]. Note that x_context is a subset of x_target.
        t_target : [traj_size, timesteps']
        x_target : Optional [traj_size, timesteps', x_dim]. Only used during training.
        context_mask: [traj_size, num_points], boolean mask specifying which element is needed to
        target_mask: [traj_size, num_points], boolean mask specifying which element is needed to
        ctx_mask_with_new_traj_obs: [traj_size, traj_size, num_points], this could represent problem setting 1
            and problem setting 2
        ctx_mask_with_new_traj_target_mask: [traj_size, traj_size, num_points]

        training: future drop out usage

        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        if training:

            _, params_rng = random.split(sample_rng, 2)
            # calculate x0
            x0 = x_context[..., 0, :]
            dummy_sampled_x0 = np.repeat(np.expand_dims(x0, 1), sample_size, axis=1)

            known_initial_cond = x_context[
                ..., 0, :
            ]  # [traj_size, state] get the first timestep, this need to assume that all the data known the initial condition!

            # extract the amortized vi's mean and std based on context data 
            mu_params_ctx, sigma_params_ctx = self.tx_to_system(
                t_context,
                x_context,
                known_initial_cond,
                ctx_mask_with_new_traj_obs,
            )  # [traj_size, z_dim]


            # extract the amortized vi's mean and std based on target data
            mu_params_tgt, sigma_params_tgt = self.tx_to_system(
                t_target,
                x_target,
                known_initial_cond,
                ctx_mask_with_new_traj_target_mask,
            )  # [traj_size, z_dim]

            # scale parameters to the ground truth range
            mu_params_ctx = self.mu_param_mapping.backward(mu_params_ctx)
            sigma_params_ctx = self.sigma_param_mapping.backward(sigma_params_ctx)
            mu_params_tgt = self.mu_param_mapping.backward(mu_params_tgt)
            sigma_params_tgt = self.sigma_param_mapping.backward(sigma_params_tgt)

            sampled_params = distributions.LogNormal(
                mu_params_tgt, sigma_params_tgt
            ).sample(
                sample_shape=(sample_size,), seed=params_rng
            )  # [sample_size, traj_size, z_dim]
            sampled_params = rearrange(
                sampled_params,
                "sample_size traj_size z_dim -> traj_size sample_size z_dim",
            )
            # regularize the parameters
            sampled_params, penalized_distance = self.param_clamping(sampled_params)
            # sampled_alpha, sampled_beta, sampled_gamma, sampled_delta = np.split(sampled_params, 4, axis=-1)
            # hack to please Flax lazy initialization: here we use this to initialize self.ode's params
            _ = self.ode(
                dummy_sampled_x0[:1], sampled_params, np.atleast_3d(t0)
            )

            # hack: we assume all the target trajectory has the same time scheduling such that we can simply make use of the first trajectory's schedule
            z_solution, ode_step = self.solve_ode(# self.solve_grey_box_ode(
                dummy_sampled_x0, self.time_scaling_coefficient * t_target[0], sampled_params, t0, self.time_scaling_coefficient * t1, solver
            )  # (timestep, traj_size, sample_size, state_dim)
            z_t_sample = rearrange(
                z_solution,
                "timestep traj_size sample_size state_dim -> traj_size sample_size timestep state_dim",
            )
            # hack: we assume all the target trajectory has the same time scheduling such that we can simply make use of the first trajectory's schedule
            y_pred_mu, y_pred_sigma = self._pi_z_to_x(
                self.time_scaling_coefficient * t_target[0], z_t_sample, sampled_params
            )  # [traj_size, sample_size, timestep, y_dim]

            return (
                y_pred_mu,
                y_pred_sigma,
                mu_params_ctx, 
                sigma_params_ctx, 
                mu_params_tgt,
                sigma_params_tgt,
                penalized_distance,
                ode_step,
            )
        
        else:
            known_initial_cond = x_context[..., 0, :]
            dummy_sampled_x0 = np.repeat(np.expand_dims(known_initial_cond, 1), sample_size, axis=1)

            _, params_rng = random.split(sample_rng, 2)

            mu_params_ctx, sigma_params_ctx = self.tx_to_system(
                t_context,
                x_context,
                known_initial_cond,
                ctx_mask_with_new_traj_obs,
            )  # [traj_size, z_dim]

            mu_params_ctx = self.mu_param_mapping.backward(mu_params_ctx)
            sigma_params_ctx = self.sigma_param_mapping.backward(sigma_params_ctx)

            sampled_params = distributions.LogNormal(
                mu_params_ctx, sigma_params_ctx
            ).sample(
                sample_shape=(sample_size,), seed=params_rng
            )  # [sample_size, traj_size, z_dim]
            sampled_params = rearrange(
                sampled_params,
                "sample_size traj_size z_dim -> traj_size sample_size z_dim",
            )
            
            # hack to please Flax lazy initialization: here we use this to initialize self.ode's params
            _ = self.ode(
                dummy_sampled_x0[:1], sampled_params, np.atleast_3d(t0)
            )

            # hack: we assume all the target trajectory has the same time scheduling such that we can simply make use of the first trajectory's schedule
            # hack: if the solution is diverging, we continue by placing np.inf on state values (and later it maybe replaced by stability_state_value_replacing)
            #   this is used because the training process can be unstable at the beginning, which is what we observed on reactor net problem
            z_solution, ode_step = self.solve_ode( # self.solve_grey_box_ode(
                dummy_sampled_x0, self.time_scaling_coefficient * t_target[0], sampled_params, t0, self.time_scaling_coefficient * t1, solver, 
                replace_exception_with_inf=True
            )  # (timestep, traj_size, sample_size, state_dim)
            z_t_sample = rearrange(
                z_solution,
                "timestep traj_size sample_size state_dim -> traj_size sample_size timestep state_dim",
            )
            # hack: we assume all the target trajectory has the same time scheduling such that we can simply make use of the first trajectory's schedule
            y_pred_mu, y_pred_sigma = self._pi_z_to_x(
                self.time_scaling_coefficient * t_target[0], z_t_sample, sampled_params
            )  # [traj_size, sample_size, timestep, y_dim]
            # Predict target points based on context
            if not self.return_param_est:
                return (
                    y_pred_mu,
                    y_pred_sigma,
                )  
            else:
                return (
                    y_pred_mu,
                    y_pred_sigma,
                    (mu_params_ctx, sigma_params_ctx,)
                )


class PI_SANODEP_LV(PI_SANODEP):
    """
    An improved version of PI-SANODEP originally implemented as Grey_Box_SANODEP_LV: we 
    do not make use of separate encoder for each of the dynamic system parameter, instead, we use a
    single encoder to encode the context data to a global representation, which largely follows the SANODEP's original format
    """
    def ode(self, z_state, params, t):
        """
        calculate the dynamics of ode: f(x) or f(x, t)
        z_state: (..., state_dim)
        t: float
        global_control: (..., state_dim)
        """
        alpha, beta, gamma, delta = params[..., :1], params[..., 1:2], params[..., 2:3], params[..., 3:]
        # calculate LV
        x, y = np.split(z_state, 2, axis=-1) 
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        return np.concatenate([dxdt, dydt], axis=-1)


class PI_SANODEP_Brusselator(PI_SANODEP):
    def ode(self, z_state, params, t):
        """
        calculate the dynamics of ode: f(x) or f(x, t)
        z_state: (..., state_dim)
        t: float
        global_control: (..., state_dim)
        """
        A, B = params[..., :1], params[..., 1:2]
        # calculate LV
        x, y = np.split(z_state, 2, axis=-1) 
        dxdt = A + x ** 2 * y - (B + 1) * x
        dydt = B * x - x ** 2 * y
        return np.concatenate([dxdt, dydt], axis=-1)


class PI_SANODEP_SELKOV(PI_SANODEP):
    def ode(self, z_state, params, t):
        """
        calculate the dynamics of ode: f(x) or f(x, t)
        z_state: (..., state_dim)
        t: float
        global_control: (..., state_dim)
        """
        a, b = params[..., :1], params[..., 1:]
        # calculate LV
        x, y = np.split(z_state, 2, axis=-1) 
        dxdt = - x + a * y + x ** 2 * y
        dydt = b - a * y - x ** 2 * y
        return np.concatenate([dxdt, dydt], axis=-1)
    

class PI_SANODEP_SIR(PI_SANODEP):
    def ode(self, z_state, params, t):
        """
        calculate the dynamics of ode: f(x) or f(x, t)
        z_state: (..., state_dim)
        t: float
        global_control: (..., state_dim)
        """
        beta, gamma = params[..., :1], params[..., 1:2]
        # calculate LV
        x, y, _ = np.split(z_state, 3, axis=-1) 
        dxdt = - beta * x * y
        dydt = beta * x * y - gamma * y
        dzdt = gamma * y
        return np.concatenate([dxdt, dydt, dzdt], axis=-1)


class PI_SANODEP_LV3D(PI_SANODEP):
    """
    An improved version of PI-SANODEP originally implemented as Grey_Box_SANODEP_LV: we 
    do not make use of separate encoder for each of the dynamic system parameter, instead, we use a
    single encoder to encode the context data to a global representation, which largely follows the SANODEP's original format
    """
    def ode(self, z_state, params, t):
        """
        calculate the dynamics of ode: f(x) or f(x, t)
        z_state: (..., state_dim)
        t: float
        global_control: (..., state_dim)
        """
        alpha, beta, epsilon, delta, gamma, zeta, eta, theta = params[..., :1], params[..., 1:2], params[..., 2:3], params[..., 3:4], params[..., 4:5], params[..., 5:6], params[..., 6:7], params[..., 7:]
        # calculate LV
        x, y, z = np.split(z_state, 3, axis=-1) 
        dxdt = alpha * x - beta * x * y - epsilon * x * z 
        dydt = delta * x * y - gamma * y - zeta * y * z
        dzdt = eta * z * x - theta * z
        return np.concatenate([dxdt, dydt, dzdt], axis=-1)


class PI_SANODEP_SIRD(PI_SANODEP):
    def ode(self, z_state, params, t):
        beta, gamma, mu = params[..., :1], params[..., 1:2], params[..., 2:]
        normed_S, normed_I, _, _ = np.split(z_state, 4, axis=-1)
        dSdt = -beta * normed_S * normed_I
        dIdt = beta * normed_S * normed_I - gamma * normed_I - mu * normed_I
        dRdt = gamma * normed_I
        dDdt = mu * normed_I
        return np.concatenate([dSdt, dIdt, dRdt, dDdt], axis=-1)
