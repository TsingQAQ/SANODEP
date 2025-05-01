"""
The interface of Neural ODE model

This module is used because putting a solver within Flax implementation based model is cumbersome to wrap with diffrax solver
refer 

Hence. this is a wrapper of the model with the additional solvers provided
"""
from flax.linen import Module, apply
from jax import random
from jax import jit
from ..models.model import NeuralODEProcess
from jax import numpy as np
import diffrax
from diffrax import PIDController, ODETerm, diffeqsolve
from einops import rearrange


from tensorflow_probability.substrates import jax as tfp
distributions = tfp.distributions


class NeuralODEWrapper:
    def __init__(self, model: Module) -> None:
        self.model = model


class NeuralODEProcessWrapper(NeuralODEWrapper):
    def __init__(self, model: NeuralODEProcess) -> None:
        assert isinstance(model, NeuralODEProcess)
        self.model = model
        self.tx_to_z_dist = lambda t, x, mask = None: jit(apply(NeuralODEProcess.tx_to_z_dist(t, x, mask)))
        self.ode = lambda z_state, global_control, t: jit(apply(NeuralODEProcess.ode(z_state, global_control, t)))
        self._z_to_x = lambda t, z, global_control: jit(apply(NeuralODEProcess._z_to_x(t, z, global_control)))

    def init(self, variables, t_context, x_context, t_target, sample_rng = None,
                 context_mask = None, target_mask = None, sample_size: int = None, 
                 x_target=None, training=None, solver='Dopri5', 
                 t0 = None, t1 = None, **kwargs):
        """
        initialize all the part (encoder, ode layer and decoder) of the model

        Args:
            t_context: [batch_size, context_size]
            x_context: [batch_size, context_size, x_dim]
            t_target: [batch_size, target_size]
            sample_rng: [batch_size,]
            context_mask: [batch_size, context_size]
            target_mask: [batch_size, target_size]
            sample_size: int
            x_target: [batch_size, target_size, x_dim]
            solver: str
            t0: float
            t1: float
        """
        (mu_z0_ctx, sigma_z0_ctx), (mu_global_ctx, sigma_global_ctx) = self.tx_to_z_dist(variables, t_context, x_context, context_mask) # (batch_size, z_dim), (batch_size, z_latent_dim)
        (mu_z0_tgt, sigma_z0_tgt), (mu_global_tgt, sigma_global_tgt) = self.tx_to_z_dist(variables, t_target, x_target, target_mask) # (batch_size, z_dim), (batch_size, z_latent_dim)
        # Sample from encoded distribution using reparameterization trick

        _, z0_rng, z_global_rng = random.split(sample_rng, 3)
        # reparam sampling the initial and control 

        # reparam sampling
        sampled_z0 = np.expand_dims(mu_z0_tgt, axis=0) + \
            np.expand_dims(sigma_z0_tgt, axis=0) * random.normal(z0_rng, shape=(sample_size, self.latent_d_dim))  # (sample_size, latent state)
        sampled_global_control = np.expand_dims(mu_global_tgt, axis=0) + \
            np.expand_dims(sigma_global_tgt, axis=0) * random.normal(z_global_rng, shape=(sample_size, self.latent_l_dim))

        # hack to init ode layers params due to jax lazy flax initialization
        # _ = self.ode(sampled_z0[:1], sampled_global_control[:1], np.atleast_2d(t0))
        def ode_f(t, z, args):
            _control = args
            return self.ode(variables, z, _control, t)
        # https://docs.kidger.site/diffrax/further_details/faq/#:~:text=See%20here.-,How%20does%20this%20compare%20to%20jax.experimental.ode.odeint%3F,-%C2%A4
        z_aug = diffeqsolve(ODETerm(ode_f), t0=t0, t1=t1, dt0=None, solver=getattr(diffrax, solver)(), 
            stepsize_controller=PIDController(rtol=1.4e-8, atol=1.4e-8), args= sampled_global_control, 
            max_steps=1, y0=sampled_z0, saveat=diffrax.SaveAt(ts=t_target)).ys
        
        z_t_sample = rearrange(z_aug, "timestep sample_size state_dim -> sample_size timestep state_dim")
        y_pred_mu, y_pred_sigma = self._z_to_x(variables, t_target, z_t_sample, sampled_global_control) # [sample_size, timestep, y_dim]

    def apply(self, variables, t_context, x_context, t_target, sample_rng = None,
                 context_mask = None, target_mask = None, sample_size: int = None, 
                 x_target=None, training=None, solver='Dopri5', 
                 t0 = None, t1 = None, **kwargs):
        """
        Args:
            t_context: [batch_size, context_size]
            x_context: [batch_size, context_size, x_dim]
            t_target: [batch_size, target_size]
            sample_rng: [batch_size,]
            context_mask: [batch_size, context_size]
            target_mask: [batch_size, target_size]
            sample_size: int
            x_target: [batch_size, target_size, x_dim]
            training: bool
            solver: str
            t0: float
            t1: float
        """
        if training:
            # Encode target and context (context needs to be encoded to calculate kl term)
            (mu_z0_ctx, sigma_z0_ctx), (mu_global_ctx, sigma_global_ctx) = self.tx_to_z_dist(variables, t_context, x_context, context_mask) # (batch_size, z_dim), (batch_size, z_latent_dim)
            (mu_z0_tgt, sigma_z0_tgt), (mu_global_tgt, sigma_global_tgt) = self.tx_to_z_dist(variables, t_target, x_target, target_mask) # (batch_size, z_dim), (batch_size, z_latent_dim)
            # Sample from encoded distribution using reparameterization trick

            _, z0_rng, z_global_rng = random.split(sample_rng, 3)
            # reparam sampling the initial and control 

            # reparam sampling
            sampled_z0 = np.expand_dims(mu_z0_tgt, axis=0) + \
                np.expand_dims(sigma_z0_tgt, axis=0) * random.normal(z0_rng, shape=(sample_size, self.latent_d_dim))  # (sample_size, latent state)
            sampled_global_control = np.expand_dims(mu_global_tgt, axis=0) + \
                np.expand_dims(sigma_global_tgt, axis=0) * random.normal(z_global_rng, shape=(sample_size, self.latent_l_dim))

            # hack to init ode layers params due to jax lazy flax initialization
            # _ = self.ode(sampled_z0[:1], sampled_global_control[:1], np.atleast_2d(t0))
            def ode_f(t, z, args):
                _control = args
                return self.ode(variables, z, _control, t)
            # https://docs.kidger.site/diffrax/further_details/faq/#:~:text=See%20here.-,How%20does%20this%20compare%20to%20jax.experimental.ode.odeint%3F,-%C2%A4
            z_aug = diffeqsolve(ODETerm(ode_f), t0=t0, t1=t1, dt0=None, solver=getattr(diffrax, solver)(), 
                stepsize_controller=PIDController(rtol=1.4e-8, atol=1.4e-8), args= sampled_global_control, 
                max_steps=1000, y0=sampled_z0, saveat=diffrax.SaveAt(ts=t_target)).ys
            
            z_t_sample = rearrange(z_aug, "timestep sample_size state_dim -> sample_size timestep state_dim")
            y_pred_mu, y_pred_sigma = self._z_to_x(variables, t_target, z_t_sample, sampled_global_control) # [sample_size, timestep, y_dim]

            return y_pred_mu, y_pred_sigma, mu_z0_ctx, sigma_z0_ctx, mu_z0_tgt, sigma_z0_tgt, mu_global_ctx, sigma_global_ctx, mu_global_tgt, sigma_global_tgt
        else:
            # At testing time, encode only context
            (mu_z0_ctx, sigma_z0_ctx), (mu_global_ctx, sigma_global_ctx) = self.tx_to_z_dist(variables, t_context, x_context, context_mask) # (z_dim), (z_latent_dim)
            # Sample from distribution based on context
            q_z0_context = distributions.MultivariateNormalDiag(mu_z0_ctx, sigma_z0_ctx)
            q_global_context = distributions.MultivariateNormalDiag(mu_global_ctx, sigma_global_ctx)
            _, z0_rng, z_global_rng = random.split(sample_rng, 3)
            sampled_z0 = q_z0_context.sample(sample_size, seed=z0_rng)
            sampled_global_control = q_global_context.sample(sample_size, z_global_rng)
            _ = self.ode(variables, sampled_z0[:1], sampled_global_control[:1], np.atleast_2d(t0))
            def ode_f(t, z, args):
                _control = args
                return self.ode(z, _control, t)
            z_aug = diffeqsolve(ODETerm(ode_f), t0=t0, t1=t1, dt0=None, solver=getattr(diffrax, solver)(), 
                stepsize_controller=PIDController(rtol=1.4e-8, atol=1.4e-8), args= sampled_global_control, 
                max_steps=1000, y0=sampled_z0, saveat=diffrax.SaveAt(ts=t_target)).ys
            
            z_t_sample = rearrange(z_aug, "timestep sample_size state_dim -> sample_size timestep state_dim")
            y_pred_mu, y_pred_sigma = self._z_to_x(variables, t_target, z_t_sample, sampled_global_control) # [sample_size, timestep, y_dim]
            # Predict target points based on context
            return y_pred_mu, y_pred_sigma    