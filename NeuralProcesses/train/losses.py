import jax
from jax import numpy as np
from jax import random
from typing import Tuple, Callable, Optional
from flax.training.train_state import TrainState
from jax.scipy.stats.norm import ppf
import sys

from tensorflow_probability.substrates import jax as tfp

distributions = tfp.distributions


def GetNeuralProcessMFVILoss(**kwargs):
    """
    Neural Processes Mean Field Variational Inference losses, the default choice used
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size

    @jax.jit
    def neural_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
        data_x, data_y, context_mask, target_mask = batch
        partial_model_apply = (
            lambda x_ctx, y_ctx, x_tgt, y_tgt, ctx_msk, tgt_msk: model.apply(
                params,
                x_context=x_ctx,
                y_context=y_ctx,
                x_target=x_tgt,
                y_target=y_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                training=training,
                context_mask=ctx_msk,
                target_mask=tgt_msk,
            )
        )

        y_pred_mu, y_pred_sigma, mu_context, sigma_context, mu_tgt, sigma_tgt = (
            jax.vmap(partial_model_apply, in_axes=0)(
                data_x, data_y, data_x, data_y, context_mask, target_mask
            )
        )
        expanded_aug_y = np.expand_dims(data_y, axis=1)
        MC_avg_target_log_likelihood = distributions.Normal(
            y_pred_mu, y_pred_sigma
        ).log_prob(expanded_aug_y) * np.expand_dims(
            np.expand_dims(target_mask, axis=1), axis=-1
        )  # [batch_size, num_samples, num_points, output_dim]
        MC_avg_target_log_likelihood = np.mean(
            np.sum(MC_avg_target_log_likelihood, axis=[-1, -2]), axis=1
        )  # [batch_size, 1]

        q_context = distributions.MultivariateNormalDiag(mu_context, sigma_context)
        q_all = distributions.MultivariateNormalDiag(mu_tgt, sigma_tgt)
        kl = distributions.kl_divergence(q_all, q_context)
        return (
            -np.mean(MC_avg_target_log_likelihood - kl),
            np.inf,
        )  # , (y_pred_mu, y_pred_sigma, mu_context, sigma_context, mu_tgt, sigma_tgt, -np.mean(MC_avg_target_log_likelihood), - np.mean(kl))

    return neural_process_mean_field_varaitional_inference_loss



def GetNeuralProcessMCLogLikelihoodLoss(**kwargs):
    """
    Implementation of the Monte Carlo based log lieklihood maximization

    detailed refer to foong2020meta
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size

    @jax.jit
    def neural_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
        data_x, data_y, context_mask, target_mask = batch
        partial_model_apply = (
            lambda x_ctx, y_ctx, x_tgt, y_tgt, ctx_msk, tgt_msk: model.apply(
                params,
                x_context=x_ctx,
                y_context=y_ctx,
                x_target=x_tgt,
                y_target=y_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                training=training,
                context_mask=ctx_msk,
                target_mask=tgt_msk,
            )
        )
        y_pred_mu, y_pred_sigma, mu_context, sigma_context, mu_tgt, sigma_tgt = (
            jax.vmap(partial_model_apply, in_axes=0)(
                data_x, data_y, data_x, data_y, context_mask, target_mask
            )
        )

        expanded_aug_y = np.expand_dims(data_y, axis=1)
        MC_log_avg_target_likelihood = distributions.Normal(
            y_pred_mu, y_pred_sigma
        ).log_prob(expanded_aug_y) * np.expand_dims(
            np.expand_dims(target_mask, axis=1), axis=-1
        )  # [batch_size, num_samples, num_points, 1]
        # note the main difference is the average is taken before the logarithm opertaion
        MC_log_avg_target_likelihood = np.squeeze(
            np.log(
                np.maximum(
                    np.mean(
                        np.exp(np.sum(MC_log_avg_target_likelihood, axis=-2)), axis=1
                    ),
                    1e-20,
                )
            ),
            axis=-1,
        )  # [batch_size, 1]

        return -np.mean(MC_log_avg_target_likelihood), np.inf

    return neural_process_mean_field_varaitional_inference_loss



def GetNeuralProcessSVIFullCovGaussianLoss(**kwargs):
    """
    Experimental Neural Processes Structural Variational Inference losses, the variational distributin is a full covariance
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size

    @jax.jit
    def neural_process_structured_varaitional_inference_full_cov_gaussian_loss(
        rng, params, batch
    ):
        
        data_x, data_y, context_mask, target_mask = batch
        partial_model_apply = (
            lambda x_ctx, y_ctx, x_tgt, y_tgt, ctx_msk, tgt_msk: model.apply(
                params,
                x_context=x_ctx,
                y_context=y_ctx,
                x_target=x_tgt,
                y_target=y_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                training=training,
                context_mask=ctx_msk,
                target_mask=tgt_msk,
            )
        )
        y_pred_mu, y_pred_sigma, mu_context, chol_context, mu_tgt, chol_tgt = jax.vmap(
            partial_model_apply, in_axes=0
        )(data_x, data_y, data_x, data_y, context_mask, target_mask)

        expanded_aug_y = np.expand_dims(data_y, axis=1)
        MC_avg_target_log_likelihood = distributions.Normal(
            y_pred_mu, y_pred_sigma
        ).log_prob(expanded_aug_y) * np.expand_dims(
            np.expand_dims(target_mask, axis=1), axis=-1
        )  # [batch_size, num_samples, num_points, 1]
        MC_avg_target_log_likelihood = np.squeeze(
            np.mean(np.sum(MC_avg_target_log_likelihood, axis=-2), axis=1), axis=-1
        )  # [batch_size, 1]

        q_context = distributions.MultivariateNormalTriL(mu_context, chol_context)
        q_all = distributions.MultivariateNormalTriL(mu_tgt, chol_tgt)
        kl = distributions.kl_divergence(q_all, q_context)
        return -np.mean(MC_avg_target_log_likelihood - kl), np.inf  # hack for steps

    return neural_process_structured_varaitional_inference_full_cov_gaussian_loss


def GetNeuralProcessTestMSELoss(**kwargs):
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size

    @jax.jit
    def neural_process_mse_loss(rng, params, batch):
        
        data_x, data_y, context_mask, target_mask = batch
        partial_model_apply = (
            lambda x_ctx, y_ctx, x_tgt, y_tgt, ctx_msk, tgt_msk: model.apply(
                params,
                x_context=x_ctx,
                y_context=y_ctx,
                x_target=x_tgt,
                y_target=y_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                training=training,
                context_mask=ctx_msk,
                target_mask=tgt_msk,
            )
        )
        y_pred_mu, _ = jax.vmap(partial_model_apply, in_axes=0)(
            data_x, data_y, data_x, data_y, context_mask, target_mask
        )

        expanded_aug_y = np.expand_dims(data_y, axis=1)
        mse = np.mean(  # average through batch
            np.sum(
                ((y_pred_mu - expanded_aug_y) ** 2)
                * np.expand_dims(np.expand_dims(target_mask, axis=1), axis=-1),
                axis=[-1, -2],
            )
            / np.expand_dims(np.sum(target_mask, axis=-1), axis=-1)
        )
        return mse, np.inf  # hack for steps

    return neural_process_mse_loss


def GetNeuralProcessNLLLoss(**kwargs):
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size

    @jax.jit
    def neural_process_nll_loss(rng, params, batch):
        
        data_x, data_y, context_mask, target_mask = batch
        partial_model_apply = (
            lambda x_ctx, y_ctx, x_tgt, y_tgt, ctx_msk, tgt_msk: model.apply(
                params,
                x_context=x_ctx,
                y_context=y_ctx,
                x_target=x_tgt,
                y_target=y_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                training=training,
                context_mask=ctx_msk,
                target_mask=tgt_msk,
            )
        )
        y_pred_mu, y_pred_sigma = jax.vmap(partial_model_apply, in_axes=0)(
            data_x, data_y, data_x, data_y, context_mask, target_mask
        )

        expanded_aug_y = np.expand_dims(data_y, axis=1)
        MC_avg_target_log_likelihood = distributions.Normal(
            y_pred_mu, y_pred_sigma
        ).log_prob(expanded_aug_y) * np.expand_dims(
            np.expand_dims(target_mask, axis=1), axis=-1
        )  # [batch_size, num_samples, num_points, 1]
        MC_avg_target_log_likelihood = np.mean(
            np.sum(MC_avg_target_log_likelihood, axis=[-1, -2]), axis=1
        )  # [batch_size]
        return -np.mean(MC_avg_target_log_likelihood), np.inf  # hack for steps

    return neural_process_nll_loss


def GetNeuralProcessEnsembleCELoss(**kwargs):
    """
    Implementation of the ensemble calibration error
    which is documented in C.12-C.17 of bootstraping neural process paper
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    levels = kwargs["config"].data.aux_cfg["CE"]["levels"] = 10
    level_ind = np.linspace(0.0, 1.0, levels + 1, endpoint=False)[1:]  # [levels]

    @jax.jit
    def neural_process_ensemble_ce_loss(rng, params, batch):
        
        data_x, data_y, context_mask, target_mask = batch
        partial_model_apply = (
            lambda x_ctx, y_ctx, x_tgt, y_tgt, ctx_msk, tgt_msk: model.apply(
                params,
                x_context=x_ctx,
                y_context=y_ctx,
                x_target=x_tgt,
                y_target=y_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                training=training,
                context_mask=ctx_msk,
                target_mask=tgt_msk,
            )
        )
        y_pred_mu, y_pred_sigma = jax.vmap(partial_model_apply, in_axes=0)(
            data_x, data_y, data_x, data_y, context_mask, target_mask
        )
        res = ppf(
            loc=y_pred_mu[..., None], scale=y_pred_sigma[..., None], q=level_ind
        )  # [batch_size, num_samples, num_points, output_dim, levels]
        aug_data_y = np.expand_dims(
            np.expand_dims(data_y, axis=1), axis=-1
        )  # [batch_size, num_samples, num_points, output_dim, levels]
        expand_target_mask = np.expand_dims(target_mask, axis=1)[..., None, None]
        p_hat = np.sum((res > aug_data_y) * expand_target_mask, axis=2) / np.sum(
            expand_target_mask, axis=-3
        )  # [batch_size, num_samples, output_dim, levels]

        p_diff = (
            p_hat - level_ind
        ) ** 2  # [batch_size, num_samples, output_dim, levels]
        p_CE = np.mean(np.sum(p_diff, axis=-1), axis=[1, -1])  # [batch_size]
        return np.mean(p_CE), np.inf  # hack for steps

    return neural_process_ensemble_ce_loss


def GetNeuralProcessSharpnessLoss(**kwargs):
    """
    Implementation of the sharpness loss
    which is documented in C.18 of bootstraping neural process paper
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size

    @jax.jit
    def neural_process_sharpness_loss(rng, params, batch):
        
        data_x, data_y, context_mask, target_mask = batch
        partial_model_apply = (
            lambda x_ctx, y_ctx, x_tgt, y_tgt, ctx_msk, tgt_msk: model.apply(
                params,
                x_context=x_ctx,
                y_context=y_ctx,
                x_target=x_tgt,
                y_target=y_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                training=training,
                context_mask=ctx_msk,
                target_mask=tgt_msk,
            )
        )
        _, y_pred_sigma = jax.vmap(partial_model_apply, in_axes=0)(
            data_x, data_y, data_x, data_y, context_mask, target_mask
        )
        expanded_target_mask = np.expand_dims(
            np.expand_dims(target_mask, axis=1), axis=-1
        )
        screened_var = (
            y_pred_sigma * expanded_target_mask
        ) ** 2  # [batch_size, num_samples, num_points, 1]
        averaged_sharpness = np.mean(
            np.sum(screened_var, axis=-2) / np.sum(expanded_target_mask, -2),
            axis=[1, 2],
        )  # [batch_size]
        return np.mean(averaged_sharpness), np.inf  # hack for steps

    return neural_process_sharpness_loss


def GetNeuralProcessMFVILossAcceptDynamicSystemData(**kwargs):
    """
    Neural Processes Mean Field Variational Inference losses, the default choice used
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)

        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt_x0, mask_ctx_x0, mask_ctx_with_new_traj: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            target_initial_cond_mask=mask_tgt_x0,
            ctx_mask_with_new_traj_obs=mask_ctx_x0,
            ctx_mask_with_new_traj_target_mask=mask_ctx_with_new_traj,
            solver="Dopri5",
            t0=t0,
            t1=t1,
        )

        (
            x_pred_mu,
            x_pred_sigma,
            mu_context,
            sigma_context,
            mu_tgt,
            sigma_tgt,
        ) = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_initial_cond_mask,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
        )

        expanded_aug_x = np.expand_dims(
            data_x, axis=2
        )  # [batch_size, traj_size, num_samples, num_points, output_dim]
        MC_avg_target_log_likelihood = distributions.Normal(
            x_pred_mu, x_pred_sigma
        ).log_prob(expanded_aug_x) * np.expand_dims(
            np.expand_dims(target_mask, axis=2), axis=-1
        )  # [batch_size, traj_size, num_samples, num_points, output_dim]

        MC_avg_target_log_likelihood = np.mean(
            np.sum(MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]

        q_context = distributions.MultivariateNormalDiag(mu_context, sigma_context)
        q_all = distributions.MultivariateNormalDiag(mu_tgt, sigma_tgt)
        kl = np.mean(
            distributions.kl_divergence(q_all, q_context), axis=-1
        )  # [batch_size]
        return (
            -np.mean(MC_avg_target_log_likelihood - kl),
            (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
        )

    return neural_process_mean_field_varaitional_inference_loss


def GetNeuralProcessTestMSEAcceptDynamicSystemData(**kwargs):
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_process_mse_loss(rng, params, batch):
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)

        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt_x0, mask_ctx_x0, mask_ctx_with_new_traj: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            target_initial_cond_mask=mask_tgt_x0,
            ctx_mask_with_new_traj_obs=mask_ctx_x0,
            ctx_mask_with_new_traj_target_mask=mask_ctx_with_new_traj,
            solver="Dopri5",
            t0=t0,
            t1=t1,
        )

        (x_pred_mu, x_pred_sigma) = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_initial_cond_mask,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
        )

        expanded_aug_x = np.expand_dims(data_x, axis=2)
        target_mask = np.diagonal(ctx_mask_with_new_traj_target_mask, axis1=-3, axis2=-2)
        element_wise_mse = ((x_pred_mu - expanded_aug_x) ** 2) * np.expand_dims(np.expand_dims(target_mask, axis=2), axis=-1)
        ml_mse = np.sum(np.mean(element_wise_mse, axis=-1),axis=-1) / np.maximum(np.expand_dims(np.sum(target_mask, axis=-1), axis=-1), 1.0)

        ml_mse_traj_mc_wise_mean = np.mean(ml_mse, axis=[-1, -2])
        return np.mean(ml_mse_traj_mc_wise_mean), (0.0, -np.mean(ml_mse_traj_mc_wise_mean), 0.0, 0.0, 0.0)

    return neural_process_mse_loss


def GetNeuralProcessTestNLLAcceptDynamicSystemData(**kwargs):
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_process_mse_loss(rng, params, batch):
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)

        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt_x0, mask_ctx_x0, mask_ctx_with_new_traj: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            target_initial_cond_mask=mask_tgt_x0,
            ctx_mask_with_new_traj_obs=mask_ctx_x0,
            ctx_mask_with_new_traj_target_mask=mask_ctx_with_new_traj,
            solver="Dopri5",
            t0=t0,
            t1=t1,
        )

        (x_pred_mu, x_pred_sigma) = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_initial_cond_mask,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
        )

        expanded_target_x = np.expand_dims(
            data_x, axis=2
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        p_x_pred = distributions.Normal(x_pred_mu, x_pred_sigma)
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, traj_size, num_samples, num_points, 1]

        target_mask = np.diagonal(ctx_mask_with_new_traj_target_mask, axis1=-3, axis2=-2)
        # use the test aligned version
        # [batch_size, traj_size, num_samples, num_points, 1]
        # target_mask: [batch_size, traj_size, num_points]
        MC_avg_target_log_likelihood = MC_avg_target_log_likelihood * \
            np.expand_dims(np.expand_dims(target_mask, axis=2), axis=-1).astype(data_x.dtype)
        MC_avg_target_log_likelihood = np.mean(
            np.sum(MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]
        return (
            -np.mean(MC_avg_target_log_likelihood),
            np.inf,
        )  # np.inf represent ode step
    

    return neural_process_mse_loss


def GetNeuralODEProcessMFVILoss(**kwargs):
    """
    Neural ODE Processes Mean Field Variational Inference losses, the defualt choice used
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
        data_t, data_x, context_mask, target_mask = batch
        # 2/14/2024 临时加的
        data_t, data_x, context_mask, target_mask = (
            data_t[:, 0, ...],
            data_x[:, 0, ...],
            context_mask[:, 0, ...],
            target_mask[:, 0, ...],
        )

        data_t = np.squeeze(data_t, axis=-1)

        sort_indices = np.argsort(data_t, axis=1)
        data_t = np.take_along_axis(data_t, sort_indices, axis=1)
        expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=1)
        context_mask = np.take_along_axis(context_mask, sort_indices, axis=1)
        target_mask = np.take_along_axis(target_mask, sort_indices, axis=1)
        batch_model_apply = (
            lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: model.apply(
                params,
                t_context=tctx,
                x_context=x_ctx,
                t_target=t_tgt,
                context_mask=mask_ctx,
                target_mask=mask_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=x_tgt,
                training=training,
                solver="Dopri5",
                t0=t0,
                t1=t1,
            )
        )

        (
            x_pred_f,
            x_pred_sigma,
            mu_z0_ctx,
            sigma_z0_ctx,
            mu_z0_tgt,
            sigma_z0_tgt,
            mu_global_ctx,
            sigma_global_ctx,
            mu_global_tgt,
            sigma_global_tgt,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t, data_x, data_t, data_x, context_mask, target_mask
        )

        # debug usage
        p_x_pred = distributions.Normal(x_pred_f, x_pred_sigma)
        q_z0_context = distributions.MultivariateNormalDiag(mu_z0_ctx, sigma_z0_ctx)
        q_z0_all = distributions.MultivariateNormalDiag(mu_z0_tgt, sigma_z0_tgt)

        q_global_context = distributions.MultivariateNormalDiag(
            mu_global_ctx, sigma_global_ctx
        )
        q_global_all = distributions.MultivariateNormalDiag(
            mu_global_tgt, sigma_global_tgt
        )
        expanded_target_x = np.expand_dims(data_x, axis=1)
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, num_samples, num_points, 1]
        masked_MC_avg_target_log_likelihood = (
            MC_avg_target_log_likelihood
            * np.expand_dims(np.expand_dims(target_mask, axis=1), axis=-1).astype(
                data_x.dtype
            )
        )
        MC_avg_target_log_likelihood = np.mean(
            np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=1
        )  # [batch_size]

        kl_z0 = distributions.kl_divergence(q_z0_all, q_z0_context)
        kl_control = distributions.kl_divergence(q_global_all, q_global_context)
        masked_form_objective = -np.mean(
            MC_avg_target_log_likelihood - (kl_z0 + kl_control)
        )
        return masked_form_objective, ode_steps

    return neural_ode_process_mean_field_varaitional_inference_loss


def GetNeuralODEProcessMFVILossAcceptMultiTrajData(**kwargs):
    """
    same Neural ODE Processes Mean Field Variational Inference losses, but accept batch of dataset
    this will only make use of the first row (i.e., first trajectory) in datset

    it is expected to be used together with NeuralODEProcessAcceptBatchData and its inherited classes
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1
    return_each_component = kwargs.get("return_each_component", False)

    @jax.jit
    def __neural_ode_process_mean_field_variational_inference_loss_component(
        rng, params, batch
    ):
        
        data_t, data_x, context_mask, target_mask = batch
        data_t = np.squeeze(data_t, axis=-1)

        sort_indices = np.argsort(
            data_t, axis=-1
        )  # [batch_size, traj_size, num_points]
        data_t = np.take_along_axis(
            data_t, sort_indices, axis=-1
        )  # [batch_size, traj_size, num_points]
        expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=-2)
        context_mask = np.take_along_axis(context_mask, sort_indices, axis=-1)
        target_mask = np.take_along_axis(target_mask, sort_indices, axis=-1)
        batch_model_apply = (
            lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: model.apply(
                params,
                t_context=tctx,
                x_context=x_ctx,
                t_target=t_tgt,
                context_mask=mask_ctx,
                target_mask=mask_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=x_tgt,
                training=training,
                solver="Dopri5",
                t0=t0,
                t1=t1,
            )
        )

        (
            x_pred_f,
            x_pred_sigma,
            mu_z0_ctx,
            sigma_z0_ctx,
            mu_z0_tgt,
            sigma_z0_tgt,
            mu_global_ctx,
            sigma_global_ctx,
            mu_global_tgt,
            sigma_global_tgt,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t, data_x, data_t, data_x, context_mask, target_mask
        )

        p_x_pred = distributions.Normal(
            x_pred_f, x_pred_sigma
        )  # [batch_size, sample_size, num_points, output_dim]
        q_z0_context = distributions.MultivariateNormalDiag(mu_z0_ctx, sigma_z0_ctx)
        q_z0_all = distributions.MultivariateNormalDiag(mu_z0_tgt, sigma_z0_tgt)

        q_global_context = distributions.MultivariateNormalDiag(
            mu_global_ctx, sigma_global_ctx
        )
        q_global_all = distributions.MultivariateNormalDiag(
            mu_global_tgt, sigma_global_tgt
        )
        # we only use the 1st data as our target data
        expanded_target_x = np.expand_dims(
            data_x[:, 0, :, :], axis=1
        )  # [batch_size, sample_size, num_points, output_dim]
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, num_samples, num_points, 1]

        #
        only_first_target_mask = target_mask[:, 0, :]  # [batch_size, num_points]
        masked_MC_avg_target_log_likelihood = (
            MC_avg_target_log_likelihood
            * np.expand_dims(
                np.expand_dims(only_first_target_mask, axis=1), axis=-1
            ).astype(data_x.dtype)
        )
        MC_avg_target_log_likelihood = np.mean(
            np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=1
        )  # [batch_size]

        kl_z0 = distributions.kl_divergence(q_z0_all, q_z0_context)
        kl_control = distributions.kl_divergence(q_global_all, q_global_context)
        return (
            np.mean(MC_avg_target_log_likelihood),
            np.mean(kl_z0),
            np.mean(kl_control),
            ode_steps,
        )

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        mc_log_likelihood, kl_z0, kl_control, ode_steps = (
            __neural_ode_process_mean_field_variational_inference_loss_component(
                rng, params, batch
            )
        )
        masked_form_objective = -(mc_log_likelihood - (kl_z0 + kl_control))
        # masked_form_objective = - np.mean(MC_avg_target_log_likelihood - (kl_z0 + kl_control))
        return masked_form_objective, ode_steps

    if not return_each_component:
        return neural_ode_process_mean_field_varaitional_inference_loss
    else:
        return __neural_ode_process_mean_field_variational_inference_loss_component


def GetBatchAugFlowMLELoss(**kwargs):
    """
    Neural ODE Processes Mean Field Variational Inference losses, the defualt choice used
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
        data_t, data_x, context_mask, target_mask = batch
        data_t = np.squeeze(data_t, axis=-1)

        sort_indices = np.argsort(
            data_t, axis=-1
        )  # [batch_size, traj_size, num_points]
        data_t = np.take_along_axis(
            data_t, sort_indices, axis=-1
        )  # [batch_size, traj_size, num_points]
        expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=-2)
        context_mask = np.take_along_axis(context_mask, sort_indices, axis=-1)
        target_mask = np.take_along_axis(target_mask, sort_indices, axis=-1)
        batch_model_apply = (
            lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: model.apply(
                params,
                t_context=tctx,
                x_context=x_ctx,
                t_target=t_tgt,
                context_mask=mask_ctx,
                target_mask=mask_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=x_tgt,
                training=training,
                solver="Dopri5",
                t0=t0,
                t1=t1,
            )
        )

        log_prob, ode_steps = jax.vmap(batch_model_apply)(
            data_t, data_x, data_t, data_x, context_mask, target_mask
        )

        only_first_target_mask = target_mask[:, 0, :]  # [batch_size, num_points]
        masked_log_prob = log_prob * np.expand_dims(
            np.expand_dims(only_first_target_mask, axis=1), axis=-1
        ).astype(
            data_x.dtype
        )  # [batch_size, sample_size, num_points, 1]

        return -np.mean(np.sum(masked_log_prob, axis=[-1, -2])), ode_steps

    return neural_ode_process_mean_field_varaitional_inference_loss


def GetBatchNeuralODEProcessX0MFVILoss(**kwargs):
    """
    Neural ODE Processes Mean Field Variational Inference losses, the defualt choice used
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
        data_t, data_x, context_mask, target_mask = batch
        data_t = np.squeeze(data_t, axis=-1)

        # sort_indices = np.argsort(data_t, axis=1)
        # data_t = np.take_along_axis(data_t, sort_indices, axis=1)
        # expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        # data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=1)
        # context_mask = np.take_along_axis(context_mask, sort_indices, axis=1)
        # target_mask = np.take_along_axis(target_mask, sort_indices, axis=1)
        sort_indices = np.argsort(
            data_t, axis=-1
        )  # [batch_size, traj_size, num_points]
        data_t = np.take_along_axis(
            data_t, sort_indices, axis=-1
        )  # [batch_size, traj_size, num_points]
        expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=-2)
        context_mask = np.take_along_axis(context_mask, sort_indices, axis=-1)
        target_mask = np.take_along_axis(target_mask, sort_indices, axis=-1)
        batch_model_apply = (
            lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: model.apply(
                params,
                t_context=tctx,
                x_context=x_ctx,
                t_target=t_tgt,
                context_mask=mask_ctx,
                target_mask=mask_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=x_tgt,
                training=training,
                solver="Dopri5",
                t0=t0,
                t1=t1,
            )
        )

        (
            x_pred_f,
            x_pred_sigma,
            mu_z0_ctx,
            sigma_z0_ctx,
            mu_z0_tgt,
            sigma_z0_tgt,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t, data_x, data_t, data_x, context_mask, target_mask
        )

        p_x_pred = distributions.Normal(
            x_pred_f, x_pred_sigma
        )  # [batch_size, sample_size, num_points, output_dim]
        q_z0_context = distributions.MultivariateNormalDiag(mu_z0_ctx, sigma_z0_ctx)
        q_z0_all = distributions.MultivariateNormalDiag(mu_z0_tgt, sigma_z0_tgt)

        # we only use the 1st data as our target data
        expanded_target_x = np.expand_dims(
            data_x[:, 0, :, :], axis=1
        )  # [batch_size, sample_size, num_points, output_dim]
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, num_samples, num_points, 1]

        #
        only_first_target_mask = target_mask[:, 0, :]  # [batch_size, num_points]
        masked_MC_avg_target_log_likelihood = (
            MC_avg_target_log_likelihood
            * np.expand_dims(
                np.expand_dims(only_first_target_mask, axis=1), axis=-1
            ).astype(data_x.dtype)
        )
        MC_avg_target_log_likelihood = np.mean(
            np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=1
        )  # [batch_size]

        kl_z0 = distributions.kl_divergence(q_z0_all, q_z0_context)
        masked_form_objective = -np.mean(MC_avg_target_log_likelihood - (kl_z0))
        return masked_form_objective, ode_steps

    return neural_ode_process_mean_field_varaitional_inference_loss


def GetBatchNeuralODEProcessStochasticGlobalMFVILoss(**kwargs):
    """
    Neural ODE Processes Mean Field Variational Inference losses, the defualt choice used
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
        data_t, data_x, context_mask, target_mask = batch
        data_t = np.squeeze(data_t, axis=-1)

        # sort_indices = np.argsort(data_t, axis=1)
        # data_t = np.take_along_axis(data_t, sort_indices, axis=1)
        # expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        # data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=1)
        # context_mask = np.take_along_axis(context_mask, sort_indices, axis=1)
        # target_mask = np.take_along_axis(target_mask, sort_indices, axis=1)
        sort_indices = np.argsort(
            data_t, axis=-1
        )  # [batch_size, traj_size, num_points]
        data_t = np.take_along_axis(
            data_t, sort_indices, axis=-1
        )  # [batch_size, traj_size, num_points]
        expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=-2)
        context_mask = np.take_along_axis(context_mask, sort_indices, axis=-1)
        target_mask = np.take_along_axis(target_mask, sort_indices, axis=-1)
        batch_model_apply = (
            lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: model.apply(
                params,
                t_context=tctx,
                x_context=x_ctx,
                t_target=t_tgt,
                context_mask=mask_ctx,
                target_mask=mask_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=x_tgt,
                training=training,
                solver="Dopri5",
                t0=t0,
                t1=t1,
            )
        )

        (
            x_pred_f,
            x_pred_sigma,
            mu_z0_ctx,
            sigma_z0_ctx,
            mu_z0_tgt,
            sigma_z0_tgt,
            mu_global_ctx,
            sigma_global_ctx,
            mu_global_tgt,
            sigma_global_tgt,
            mu_g_dynamic_ctx,
            sigma_g_dynamic_ctx,
            mu_g_dynamic_tgt,
            sigma_g_dynamic_tgt,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t, data_x, data_t, data_x, context_mask, target_mask
        )

        p_x_pred = distributions.Normal(
            x_pred_f, x_pred_sigma
        )  # [batch_size, sample_size, num_points, output_dim]
        q_z0_context = distributions.MultivariateNormalDiag(mu_z0_ctx, sigma_z0_ctx)
        q_z0_all = distributions.MultivariateNormalDiag(mu_z0_tgt, sigma_z0_tgt)

        q_global_context = distributions.MultivariateNormalDiag(
            mu_global_ctx, sigma_global_ctx
        )
        q_global_all = distributions.MultivariateNormalDiag(
            mu_global_tgt, sigma_global_tgt
        )

        q_g_dynamic_context = distributions.MultivariateNormalDiag(
            mu_g_dynamic_ctx, sigma_g_dynamic_ctx
        )
        q_g_dynamic_all = distributions.MultivariateNormalDiag(
            mu_g_dynamic_tgt, sigma_g_dynamic_tgt
        )
        # we only use the 1st data as our target data
        expanded_target_x = np.expand_dims(
            data_x[:, 0, :, :], axis=1
        )  # [batch_size, sample_size, num_points, output_dim]
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, num_samples, num_points, 1]

        #
        only_first_target_mask = target_mask[:, 0, :]  # [batch_size, num_points]
        masked_MC_avg_target_log_likelihood = (
            MC_avg_target_log_likelihood
            * np.expand_dims(
                np.expand_dims(only_first_target_mask, axis=1), axis=-1
            ).astype(data_x.dtype)
        )
        MC_avg_target_log_likelihood = np.mean(
            np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=1
        )  # [batch_size]

        kl_z0 = distributions.kl_divergence(q_z0_all, q_z0_context)
        kl_control = distributions.kl_divergence(q_global_all, q_global_context)
        kl_g_dynamic = distributions.kl_divergence(q_g_dynamic_all, q_g_dynamic_context)
        masked_form_objective = -np.mean(
            MC_avg_target_log_likelihood - (kl_z0 + kl_control + kl_g_dynamic)
        )
        return masked_form_objective, ode_steps

    return neural_ode_process_mean_field_varaitional_inference_loss


def GetNeuralODEProcessMFVIMultiTrajectoryAwareLoss(**kwargs):
    """
    Neural ODE Processes Mean Field Variational Inference losses, the defualt choice used

    the loss is calculated by averaging multiple trajectories
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1
 
    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)
        context_mask  = np.einsum('biih->bih', ctx_mask_with_new_traj_obs) 
        target_mask = np.einsum('biih->bih', ctx_mask_with_new_traj_target_mask)
        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt, mask_ctx: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            context_mask = mask_ctx, 
            target_mask = mask_tgt,
            solver="Dopri5",
            t0=t0,
            t1=t1)
        (
            x_pred_f,
            x_pred_sigma,
            mu_z0_ctx, 
            sigma_z0_ctx, 
            mu_z0_tgt,
            sigma_z0_tgt,
            mu_global_ctx,
            sigma_global_ctx,
            mu_global_tgt,
            sigma_global_tgt,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_mask,
            context_mask,
        )

        p_x_pred = distributions.Normal(
            x_pred_f, x_pred_sigma
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        q_z0_context = distributions.MultivariateNormalDiag(mu_z0_ctx, sigma_z0_ctx)
        q_z0_all = distributions.MultivariateNormalDiag(mu_z0_tgt, sigma_z0_tgt)

        q_global_context = distributions.MultivariateNormalDiag(
            mu_global_ctx, sigma_global_ctx
        )
        q_global_all = distributions.MultivariateNormalDiag(
            mu_global_tgt, sigma_global_tgt
        )

        # we only use the 1st data as our target data
        expanded_target_x = np.expand_dims(
            data_x, 2
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, traj_size, num_samples, num_points, 1]

        masked_MC_avg_target_log_likelihood = (
            MC_avg_target_log_likelihood
            * np.expand_dims(
                np.expand_dims(target_mask, axis=2), axis=-1
            ).astype(data_x.dtype)
        )
        MC_avg_target_log_likelihood = np.mean(
            np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]

        kl_z0 = np.mean(distributions.kl_divergence(q_z0_all, q_z0_context), axis=-1)
        kl_control = np.mean(
            distributions.kl_divergence(q_global_all, q_global_context), axis=-1
        )
        masked_form_objective = -np.mean(
            MC_avg_target_log_likelihood - (kl_z0 + kl_control)
        )
        return masked_form_objective, (
            ode_steps,
            -np.mean(MC_avg_target_log_likelihood),
            np.mean(kl_z0),
            np.mean(kl_control),
            0.0,
        )

    return neural_ode_process_mean_field_varaitional_inference_loss


def GetNeuralODEProcessMFVILossAcceptMultiTrajDataWithStochasticGlobalMFVILoss(
    **kwargs,
):
    """
    Neural ODE Processes Mean Field Variational Inference losses, the defualt choice used

    the loss is calculated by averaging multiple trajectories
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
        data_t, data_x, context_mask, target_mask = batch
        data_t = np.squeeze(data_t, axis=-1)

        # sort_indices = np.argsort(data_t, axis=1)
        # data_t = np.take_along_axis(data_t, sort_indices, axis=1)
        # expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        # data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=1)
        # context_mask = np.take_along_axis(context_mask, sort_indices, axis=1)
        # target_mask = np.take_along_axis(target_mask, sort_indices, axis=1)
        sort_indices = np.argsort(
            data_t, axis=-1
        )  # [batch_size, traj_size, num_points]
        data_t = np.take_along_axis(
            data_t, sort_indices, axis=-1
        )  # [batch_size, traj_size, num_points]
        expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=-2)
        context_mask = np.take_along_axis(context_mask, sort_indices, axis=-1)
        target_mask = np.take_along_axis(target_mask, sort_indices, axis=-1)
        batch_model_apply = (
            lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: model.apply(
                params,
                t_context=tctx,
                x_context=x_ctx,
                t_target=t_tgt,
                context_mask=mask_ctx,
                target_mask=mask_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=x_tgt,
                training=training,
                solver="Dopri5",
                t0=t0,
                t1=t1,
            )
        )

        (
            x_pred_f,
            x_pred_sigma,
            mu_z0_ctx,
            sigma_z0_ctx,
            mu_z0_tgt,
            sigma_z0_tgt,
            mu_global_ctx,
            sigma_global_ctx,
            mu_global_tgt,
            sigma_global_tgt,
            mu_g_dynamic_ctx,
            sigma_g_dynamic_ctx,
            mu_g_dynamic_tgt,
            sigma_g_dynamic_tgt,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t, data_x, data_t, data_x, context_mask, target_mask
        )

        p_x_pred = distributions.Normal(
            x_pred_f, x_pred_sigma
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        q_z0_context = distributions.MultivariateNormalDiag(mu_z0_ctx, sigma_z0_ctx)
        q_z0_all = distributions.MultivariateNormalDiag(mu_z0_tgt, sigma_z0_tgt)

        q_global_context = distributions.MultivariateNormalDiag(
            mu_global_ctx, sigma_global_ctx
        )
        q_global_all = distributions.MultivariateNormalDiag(
            mu_global_tgt, sigma_global_tgt
        )

        q_g_dynamic_context = distributions.MultivariateNormalDiag(
            mu_g_dynamic_ctx, sigma_g_dynamic_ctx
        )
        q_g_dynamic_all = distributions.MultivariateNormalDiag(
            mu_g_dynamic_tgt, sigma_g_dynamic_tgt
        )
        # we only use the 1st data as our target data
        expanded_target_x = np.expand_dims(
            data_x, 2
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, traj_size, num_samples, num_points, 1]

        only_first_target_mask = target_mask  # [batch_size, traj_size, num_points]
        masked_MC_avg_target_log_likelihood = (
            MC_avg_target_log_likelihood
            * np.expand_dims(
                np.expand_dims(only_first_target_mask, axis=2), axis=-1
            ).astype(data_x.dtype)
        )
        MC_avg_target_log_likelihood = np.mean(
            np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]

        kl_z0 = np.mean(distributions.kl_divergence(q_z0_all, q_z0_context), axis=-1)
        kl_control = np.mean(
            distributions.kl_divergence(q_global_all, q_global_context), axis=-1
        )
        kl_g_dynamic = distributions.kl_divergence(q_g_dynamic_all, q_g_dynamic_context)
        masked_form_objective = -np.mean(
            MC_avg_target_log_likelihood - (kl_z0 + kl_control + kl_g_dynamic)
        )
        return masked_form_objective, ode_steps

    return neural_ode_process_mean_field_varaitional_inference_loss


def GetNeuralODEProcessMFVILossAcceptMultiTrajDataWithCorrectStochasticGlobalMFVILoss(
    **kwargs,
):
    """
    Neural ODE Processes Mean Field Variational Inference losses, the defualt choice used

    the loss is calculated by averaging multiple trajectories
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
        data_t, data_x, context_mask, target_mask = batch
        data_t = np.squeeze(data_t, axis=-1)

        # sort_indices = np.argsort(data_t, axis=1)
        # data_t = np.take_along_axis(data_t, sort_indices, axis=1)
        # expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        # data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=1)
        # context_mask = np.take_along_axis(context_mask, sort_indices, axis=1)
        # target_mask = np.take_along_axis(target_mask, sort_indices, axis=1)
        sort_indices = np.argsort(
            data_t, axis=-1
        )  # [batch_size, traj_size, num_points]
        data_t = np.take_along_axis(
            data_t, sort_indices, axis=-1
        )  # [batch_size, traj_size, num_points]
        expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=-2)
        context_mask = np.take_along_axis(context_mask, sort_indices, axis=-1)
        target_mask = np.take_along_axis(target_mask, sort_indices, axis=-1)
        batch_model_apply = (
            lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: model.apply(
                params,
                t_context=tctx,
                x_context=x_ctx,
                t_target=t_tgt,
                context_mask=mask_ctx,
                target_mask=mask_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=x_tgt,
                training=training,
                solver="Dopri5",
                t0=t0,
                t1=t1,
            )
        )

        (
            x_pred_f,
            x_pred_sigma,
            mu_z0_ctx,
            sigma_z0_ctx,
            mu_z0_tgt,
            sigma_z0_tgt,
            mu_global_ctx,
            sigma_global_ctx,
            mu_global_tgt,
            sigma_global_tgt,
            mu_g_dynamic_ctx,
            sigma_g_dynamic_ctx,
            mu_g_dynamic_tgt,
            sigma_g_dynamic_tgt,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t, data_x, data_t, data_x, context_mask, target_mask
        )

        p_x_pred = distributions.Normal(
            x_pred_f, x_pred_sigma
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        q_z0_context = distributions.MultivariateNormalDiag(mu_z0_ctx, sigma_z0_ctx)
        q_z0_all = distributions.MultivariateNormalDiag(mu_z0_tgt, sigma_z0_tgt)

        q_global_context = distributions.MultivariateNormalDiag(
            mu_global_ctx, sigma_global_ctx
        )
        q_global_all = distributions.MultivariateNormalDiag(
            mu_global_tgt, sigma_global_tgt
        )

        q_g_dynamic_context = distributions.MultivariateNormalDiag(
            mu_g_dynamic_ctx, sigma_g_dynamic_ctx
        )  # [batch_size, sample_size, dim]
        q_g_dynamic_all = distributions.MultivariateNormalDiag(
            mu_g_dynamic_tgt, sigma_g_dynamic_tgt
        )  # [batch_size, sample_size, dim]
        # we only use the 1st data as our target data
        expanded_target_x = np.expand_dims(
            data_x, 2
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, traj_size, num_samples, num_points, 1]

        only_first_target_mask = target_mask  # [batch_size, traj_size, num_points]
        masked_MC_avg_target_log_likelihood = (
            MC_avg_target_log_likelihood
            * np.expand_dims(
                np.expand_dims(only_first_target_mask, axis=2), axis=-1
            ).astype(data_x.dtype)
        )
        MC_avg_target_log_likelihood = np.mean(
            np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]

        kl_z0 = np.mean(distributions.kl_divergence(q_z0_all, q_z0_context), axis=-1)
        kl_control = np.mean(
            distributions.kl_divergence(q_global_all, q_global_context), axis=-1
        )
        kl_g_dynamic = np.mean(
            distributions.kl_divergence(q_g_dynamic_all, q_g_dynamic_context), axis=-1
        )
        masked_form_objective = -np.mean(
            MC_avg_target_log_likelihood - (kl_z0 + kl_control + kl_g_dynamic)
        )
        return masked_form_objective, ode_steps

    return neural_ode_process_mean_field_varaitional_inference_loss



def GetSANODEPTestMSELossAcceptBatchData(**kwargs):
    """
    the loss is calculated by averaging multiple trajectories in a Monte Carlo fashion
    the loss is calculated by considering two problem settings
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mse_loss(rng, params, batch):
        # 
        # data_t, data_x, known_trajectory, context_mask_pc1, context_mask_pc2, target_mask = batch
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)

        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt_x0, mask_ctx_x0, mask_ctx_with_new_traj: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            target_initial_cond_mask=mask_tgt_x0,
            ctx_mask_with_new_traj_obs=mask_ctx_x0,
            ctx_mask_with_new_traj_target_mask=mask_ctx_with_new_traj,
            solver="Dopri5",
            t0=t0,
            t1=t1,
        )

        (
            x_pred_f,
            _,
        ) = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_initial_cond_mask,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
        )

        # we follow neural ode process by calculating mse per each sample and then average over the samples
        expanded_aug_x = np.expand_dims(data_x, axis=2)
        target_mask = np.diagonal(ctx_mask_with_new_traj_target_mask, axis1=-3, axis2=-2)
        element_wise_mse = ((x_pred_f - expanded_aug_x) ** 2) * np.expand_dims(np.expand_dims(target_mask, axis=2), axis=-1)
        ml_mse = np.sum(np.mean(element_wise_mse, axis=-1),axis=-1) / np.maximum(np.expand_dims(np.sum(target_mask, axis=-1), axis=-1), 1.0)

        ml_mse_traj_mc_wise_mean = np.mean(ml_mse, axis=[-1, -2])
        return np.mean(ml_mse_traj_mc_wise_mean), -np.mean(ml_mse_traj_mc_wise_mean)

    return neural_ode_process_mse_loss


def GetSANODEPNLLLossAcceptBatchData(**kwargs):
    """
    the loss is calculated by averaging multiple trajectories in a Monte Carlo fashion
    the loss is calculated by considering two problem settings
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_nll_loss(rng, params, batch):
        # 
        # data_t, data_x, known_trajectory, context_mask_pc1, context_mask_pc2, target_mask = batch
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)

        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt_x0, mask_ctx_x0, mask_ctx_with_new_traj: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            target_initial_cond_mask=mask_tgt_x0,
            ctx_mask_with_new_traj_obs=mask_ctx_x0,
            ctx_mask_with_new_traj_target_mask=mask_ctx_with_new_traj,
            solver="Dopri5",
            t0=t0,
            t1=t1,
        )

        # x_pred_f, x_pred_sigma = jax.vmap(batch_model_apply)(data_t, data_x, data_t, data_x, context_mask, target_mask)
        x_pred_f, x_pred_sigma = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_initial_cond_mask,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
        )
        expanded_target_x = np.expand_dims(
            data_x, axis=2
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        p_x_pred = distributions.Normal(x_pred_f, x_pred_sigma)
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, traj_size, num_samples, num_points, 1]
        target_mask = np.diagonal(ctx_mask_with_new_traj_target_mask, axis1=-3, axis2=-2)
        # use the test aligned version
        # [batch_size, traj_size, num_samples, num_points, 1]
        # target_mask: [batch_size, traj_size, num_points]
        MC_avg_target_log_likelihood = MC_avg_target_log_likelihood * \
            np.expand_dims(np.expand_dims(target_mask, axis=2), axis=-1).astype(data_x.dtype)
        MC_avg_target_log_likelihood = np.mean(
            np.sum(MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]
        return -np.mean(MC_avg_target_log_likelihood), np.inf,# np.inf represent ode step
    
        # masked_MC_avg_target_log_likelihood = (
        #     MC_avg_target_log_likelihood
        #     * np.expand_dims(
        #         np.expand_dims(target_mask_unknown_traj, axis=2), axis=-1
        #     ).astype(data_x.dtype)
        # )

        # Here we cannot do mean anymore because we have screened out the known trajectory
        # MC_avg_target_log_likelihood = np.sum(
        #     np.mean(
        #         np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=-1
        #     ),
        #     axis=-1,
        # ) / (
        #     traj_size - known_trajectory
        # )  # [batch_size]

        # masked_MC_avg_target_log_likelihood = MC_avg_target_log_likelihood * np.expand_dims(  # expand with the sample_size dimension as well as the states dimension
        #     np.expand_dims(target_mask, axis=2), axis=-1
        # ).astype(
        #     data_x.dtype
        # )
        # # sum log likelihood and average through sample size and MC trajectories
        # MC_avg_target_log_likelihood = np.mean(
        #     np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        # )  # [batch_size]
        # MC_avg_target_log_likelihood = np.mean(
        #     np.sum(np.mean(masked_MC_avg_target_log_likelihood, axis=-1), axis=-1) / 
        #     np.maximum(np.expand_dims(np.sum(target_mask, axis=-1), axis=-1), 1.0), axis=[-1, -2]
        # )  # [batch_size]

        # return -np.mean(MC_avg_target_log_likelihood), np.inf
        # # expanded_aug_x = np.expand_dims(data_x, axis=1)
        # expanded_aug_x = np.expand_dims(data_x[:, 0, :, :], axis=1)
        # # mse = np.mean(np.sum(((x_pred_f - expanded_aug_x) ** 2) * np.expand_dims(np.expand_dims(target_mask, axis=1), axis=-1), axis=-2))
        # only_first_target_mask = target_mask[:, 0, :]
        # MC_avg_target_log_likelihood = distributions.Normal(x_pred_f, x_pred_sigma).log_prob(expanded_aug_x) *\
        #     np.expand_dims(np.expand_dims(only_first_target_mask, axis=1), axis=-1) # [batch_size, num_samples, num_points, 1]
        # MC_avg_target_log_likelihood = np.mean(np.sum(MC_avg_target_log_likelihood, axis=[-1, -2]), axis=1) # [batch_size]
        # return - np.mean(MC_avg_target_log_likelihood), np.inf # np.inf represent ode step

    return neural_ode_process_nll_loss


def GetNeuralODEProcessMFVILossUncondz0ConddsysLoss(**kwargs):
    """
    Problem setting 1 corresponding loss:
    here we use prior z0 and posterior dsys, which is similar as Jiang 2022
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)

        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt_x0, mask_ctx_x0, mask_ctx_with_new_traj: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            target_initial_cond_mask=mask_tgt_x0,
            ctx_mask_with_new_traj_obs=mask_ctx_x0,
            ctx_mask_with_new_traj_target_mask=mask_ctx_with_new_traj,
            solver="Dopri5",
            t0=t0,
            t1=t1,
        )

        (
            x_pred_f,
            x_pred_sigma,
            mu_z0_tgt,
            sigma_z0_tgt,
            mu_sys_dynamic_ctx,
            sigma_sys_dynamic_ctx,
            mu_sys_dynamic_tgt,
            sigma_sys_dynamic_tgt,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_initial_cond_mask,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
        )
        # this does not need to modify
        p_x_pred = distributions.Normal(
            x_pred_f, x_pred_sigma
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]

        q_z0_tgt = distributions.MultivariateNormalDiag(mu_z0_tgt, sigma_z0_tgt)

        # This line of code follows Jiang et al 2022, we may want to change here!
        q_z0_tgt_prior = distributions.MultivariateNormalDiag(
            np.zeros_like(mu_z0_tgt), np.ones_like(sigma_z0_tgt)
        )

        q_g_dynamic_context = distributions.MultivariateNormalDiag(
            mu_sys_dynamic_ctx, sigma_sys_dynamic_ctx
        )  # [batch_size, traj_size, dim]
        q_g_dynamic_all = distributions.MultivariateNormalDiag(
            mu_sys_dynamic_tgt, sigma_sys_dynamic_tgt
        )  # [batch_size, traj_size, dim]

        # calculate the vi sampled log likelihood on the target data
        # we only use the 1st data as our target data
        expanded_target_x = np.expand_dims(
            data_x, axis=2
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, traj_size, num_samples, num_points, 1]

        masked_MC_avg_target_log_likelihood = MC_avg_target_log_likelihood * np.expand_dims(  # expand with the sample_size dimension as well as the states dimension
            np.expand_dims(target_mask, axis=2), axis=-1
        ).astype(
            data_x.dtype
        )
        # sum log likelihood and average through sample size and MC trajectories
        MC_avg_target_log_likelihood = np.mean(
            np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]

        kl_g_dynamic = np.mean(
            distributions.kl_divergence(q_g_dynamic_all, q_g_dynamic_context), axis=-1
        )
        kl_z0_tgt = np.mean(
            distributions.kl_divergence(q_z0_tgt, q_z0_tgt_prior), axis=-1
        )

        masked_form_objective = -np.mean(
            MC_avg_target_log_likelihood - (kl_g_dynamic + kl_z0_tgt)
        )

        # note that all the rest are used for supervising the training process
        return masked_form_objective, (
            ode_steps,
            -np.mean(MC_avg_target_log_likelihood),
            0.0,
            np.mean(kl_g_dynamic),
            np.mean(kl_z0_tgt),
        )

    return neural_ode_process_mean_field_varaitional_inference_loss


def GetNeuralODEProcessMFVILossCondz0ConddsysLoss(**kwargs):
    """
    Problem setting 1 corresponding loss:
    here we use posterior z0 and posterior dsys
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)

        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt_x0, mask_ctx_x0, mask_ctx_with_new_traj: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            target_initial_cond_mask=mask_tgt_x0,
            ctx_mask_with_new_traj_obs=mask_ctx_x0,
            ctx_mask_with_new_traj_target_mask=mask_ctx_with_new_traj,
            solver="Dopri5",
            t0=t0,
            t1=t1,
        )
        (
            x_pred_f,
            x_pred_sigma,
            mu_z0_tgt,
            sigma_z0_tgt,
            mu_sys_dynamic_ctx,
            sigma_sys_dynamic_ctx,
            mu_sys_dynamic_tgt,
            sigma_sys_dynamic_tgt,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_initial_cond_mask,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
        )
        # this does not need to modify
        p_x_pred = distributions.Normal(
            x_pred_f, x_pred_sigma
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]

        q_g_dynamic_context = distributions.MultivariateNormalDiag(
            mu_sys_dynamic_ctx, sigma_sys_dynamic_ctx
        )  # [batch_size, traj_size, dim]
        q_g_dynamic_all = distributions.MultivariateNormalDiag(
            mu_sys_dynamic_tgt, sigma_sys_dynamic_tgt
        )  # [batch_size, traj_size, dim]

        # calculate the vi sampled log likelihood on the target data
        # we only use the 1st data as our target data
        expanded_target_x = np.expand_dims(
            data_x, axis=2
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, traj_size, num_samples, num_points, 1]

        masked_MC_avg_target_log_likelihood = MC_avg_target_log_likelihood * np.expand_dims(  # expand with the sample_size dimension as well as the states dimension
            np.expand_dims(target_mask, axis=2), axis=-1
        ).astype(
            data_x.dtype
        )
        # sum log likelihood and average through sample size and MC trajectories
        MC_avg_target_log_likelihood = np.mean(
            np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]

        kl_g_dynamic = np.mean(
            distributions.kl_divergence(q_g_dynamic_all, q_g_dynamic_context), axis=-1
        )

        masked_form_objective = -np.mean(MC_avg_target_log_likelihood - (kl_g_dynamic))

        # note that all the rest are used for supervising the training process
        return masked_form_objective, (
            ode_steps,
            -np.mean(MC_avg_target_log_likelihood),
            0.0,
            np.mean(kl_g_dynamic),
            0.0,
        )

    return neural_ode_process_mean_field_varaitional_inference_loss


def GetNeuralODEProcessV1MFVILossCondz0ConddsysLoss(**kwargs):
    """
    Problem setting 1 corresponding loss:
    here we use posterior z0 and posterior dsys
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)

        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt_x0, mask_ctx_x0, mask_ctx_with_new_traj: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            target_initial_cond_mask=mask_tgt_x0,
            ctx_mask_with_new_traj_obs=mask_ctx_x0,
            ctx_mask_with_new_traj_target_mask=mask_ctx_with_new_traj,
            solver="Dopri5",
            t0=t0,
            t1=t1,
        )
        (
            x_pred_f,
            x_pred_sigma,
            mu_z0_tgt,
            sigma_z0_tgt,
            mu_z0_ctx, 
            sigma_z0_ctx,
            mu_sys_dynamic_ctx,
            sigma_sys_dynamic_ctx,
            mu_sys_dynamic_tgt,
            sigma_sys_dynamic_tgt,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_initial_cond_mask,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
        )
        # this does not need to modify
        p_x_pred = distributions.Normal(
            x_pred_f, x_pred_sigma
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]

        q_z0_context = distributions.MultivariateNormalDiag(mu_z0_ctx, sigma_z0_ctx)
        q_z0_target = distributions.MultivariateNormalDiag(mu_z0_tgt, sigma_z0_tgt)
        q_g_dynamic_context = distributions.MultivariateNormalDiag(
            mu_sys_dynamic_ctx, sigma_sys_dynamic_ctx
        )  # [batch_size, traj_size, dim]
        q_g_dynamic_all = distributions.MultivariateNormalDiag(
            mu_sys_dynamic_tgt, sigma_sys_dynamic_tgt
        )  # [batch_size, traj_size, dim]

        # calculate the vi sampled log likelihood on the target data
        # we only use the 1st data as our target data
        expanded_target_x = np.expand_dims(
            data_x, axis=2
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, traj_size, num_samples, num_points, 1]

        masked_MC_avg_target_log_likelihood = MC_avg_target_log_likelihood * np.expand_dims(  # expand with the sample_size dimension as well as the states dimension
            np.expand_dims(target_mask, axis=2), axis=-1
        ).astype(
            data_x.dtype
        )
        # sum log likelihood and average through sample size and MC trajectories
        MC_avg_target_log_likelihood = np.mean(
            np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]

        kl_g_dynamic = np.mean(
            distributions.kl_divergence(q_g_dynamic_all, q_g_dynamic_context), axis=-1
        )
        kl_z0 = np.mean(
            distributions.kl_divergence(q_z0_target, q_z0_context), axis=-1
        )

        masked_form_objective = -np.mean(MC_avg_target_log_likelihood - (kl_g_dynamic + kl_z0))

        # note that all the rest are used for supervising the training process
        return masked_form_objective, (
            ode_steps,
            -np.mean(MC_avg_target_log_likelihood),
            np.mean(kl_z0),
            np.mean(kl_g_dynamic),
            0.0,
        )

    return neural_ode_process_mean_field_varaitional_inference_loss


def GetNeuralODEProcessV1MFVILossUncondz0ConddsysLoss(**kwargs):
    """
    Problem setting 1 corresponding loss:
    here we use posterior z0 and posterior dsys
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)

        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt_x0, mask_ctx_x0, mask_ctx_with_new_traj: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            target_initial_cond_mask=mask_tgt_x0,
            ctx_mask_with_new_traj_obs=mask_ctx_x0,
            ctx_mask_with_new_traj_target_mask=mask_ctx_with_new_traj,
            solver="Dopri5",
            t0=t0,
            t1=t1,
        )
        (
            x_pred_f,
            x_pred_sigma,
            mu_z0_tgt,
            sigma_z0_tgt,
            mu_z0_ctx, 
            sigma_z0_ctx,
            mu_sys_dynamic_ctx,
            sigma_sys_dynamic_ctx,
            mu_sys_dynamic_tgt,
            sigma_sys_dynamic_tgt,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_initial_cond_mask,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
        )
        # this does not need to modify
        p_x_pred = distributions.Normal(
            x_pred_f, x_pred_sigma
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        q_z0_ctx = distributions.MultivariateNormalDiag(
            np.zeros_like(mu_z0_tgt), np.ones_like(sigma_z0_tgt)
        )
        q_z0_target = distributions.MultivariateNormalDiag(mu_z0_tgt, sigma_z0_tgt)
        q_g_dynamic_context = distributions.MultivariateNormalDiag(
            mu_sys_dynamic_ctx, sigma_sys_dynamic_ctx
        )  # [batch_size, traj_size, dim]
        q_g_dynamic_all = distributions.MultivariateNormalDiag(
            mu_sys_dynamic_tgt, sigma_sys_dynamic_tgt
        )  # [batch_size, traj_size, dim]

        # calculate the vi sampled log likelihood on the target data
        # we only use the 1st data as our target data
        expanded_target_x = np.expand_dims(
            data_x, axis=2
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, traj_size, num_samples, num_points, 1]

        masked_MC_avg_target_log_likelihood = MC_avg_target_log_likelihood * np.expand_dims(  # expand with the sample_size dimension as well as the states dimension
            np.expand_dims(target_mask, axis=2), axis=-1
        ).astype(
            data_x.dtype
        )
        # sum log likelihood and average through sample size and MC trajectories
        MC_avg_target_log_likelihood = np.mean(
            np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]

        kl_g_dynamic = np.mean(
            distributions.kl_divergence(q_g_dynamic_all, q_g_dynamic_context), axis=-1
        )
        kl_z0 = np.mean(
            distributions.kl_divergence(q_z0_target, q_z0_ctx), axis=-1
        )

        masked_form_objective = -np.mean(MC_avg_target_log_likelihood - (kl_g_dynamic + kl_z0))

        # note that all the rest are used for supervising the training process
        return masked_form_objective, (
            ode_steps,
            -np.mean(MC_avg_target_log_likelihood),
            0.0, 
            np.mean(kl_g_dynamic),
            np.mean(kl_z0),
        )

    return neural_ode_process_mean_field_varaitional_inference_loss


def GetNeuralODEProcessMFVILossCondz0UconddsysLoss(**kwargs):
    """
    Problem setting 1 corresponding loss:
    here we use posterior z0 and prior dsys
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)

        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt_x0, mask_ctx_x0, mask_ctx_with_new_traj: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            target_initial_cond_mask=mask_tgt_x0,
            ctx_mask_with_new_traj_obs=mask_ctx_x0,
            ctx_mask_with_new_traj_target_mask=mask_ctx_with_new_traj,
            solver="Dopri5",
            t0=t0,
            t1=t1,
        )

        (
            x_pred_f,
            x_pred_sigma,
            mu_z0_tgt,
            sigma_z0_tgt,
            mu_sys_dynamic_ctx,
            sigma_sys_dynamic_ctx,
            mu_sys_dynamic_tgt,
            sigma_sys_dynamic_tgt,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_initial_cond_mask,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
        )
        # this does not need to modify
        p_x_pred = distributions.Normal(
            x_pred_f, x_pred_sigma
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]

        q_g_dynamic_all = distributions.MultivariateNormalDiag(
            mu_sys_dynamic_tgt, sigma_sys_dynamic_tgt
        )  # [batch_size, traj_size, dim]
        q_g_dynamic_prior = distributions.MultivariateNormalDiag(
            np.zeros_like(mu_sys_dynamic_tgt), np.ones_like(sigma_sys_dynamic_tgt)
        )

        # calculate the vi sampled log likelihood on the target data
        # we only use the 1st data as our target data
        expanded_target_x = np.expand_dims(
            data_x, axis=2
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, traj_size, num_samples, num_points, 1]

        masked_MC_avg_target_log_likelihood = MC_avg_target_log_likelihood * np.expand_dims(  # expand with the sample_size dimension as well as the states dimension
            np.expand_dims(target_mask, axis=2), axis=-1
        ).astype(
            data_x.dtype
        )
        # sum log likelihood and average through sample size and MC trajectories
        MC_avg_target_log_likelihood = np.mean(
            np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]

        kl_g_dynamic = np.mean(
            distributions.kl_divergence(q_g_dynamic_all, q_g_dynamic_prior), axis=-1
        )

        masked_form_objective = -np.mean(MC_avg_target_log_likelihood - (kl_g_dynamic))

        # note that all the rest are used for supervising the training process
        return masked_form_objective, (
            ode_steps,
            -np.mean(MC_avg_target_log_likelihood),
            0.0,
            np.mean(kl_g_dynamic),
            0.0,
        )

    return neural_ode_process_mean_field_varaitional_inference_loss


def GetNeuralODEProcessMFVILossUncondz0UnconddsysLoss(**kwargs):
    """
    Problem setting 1 corresponding loss:
    here we use prior z0 and posterior dsys, which is similar as Jiang 2022
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1
    # pred_initial_weights = kwargs['config'].training.pred_initial_weights

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)

        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt_x0, mask_ctx_x0, mask_ctx_with_new_traj: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            target_initial_cond_mask=mask_tgt_x0,
            ctx_mask_with_new_traj_obs=mask_ctx_x0,
            ctx_mask_with_new_traj_target_mask=mask_ctx_with_new_traj,
            solver="Dopri5",
            t0=t0,
            t1=t1,
        )
        (
            x_pred_f,
            x_pred_sigma,
            mu_z0_tgt,
            sigma_z0_tgt,
            mu_sys_dynamic_ctx,
            sigma_sys_dynamic_ctx,
            mu_sys_dynamic_tgt,
            sigma_sys_dynamic_tgt,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_initial_cond_mask,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
        )
        # this does not need to modify
        p_x_pred = distributions.Normal(
            x_pred_f, x_pred_sigma
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]

        # q_z0_context = distributions.MultivariateNormalDiag(mu_z0_ctx_ps1, sigma_z0_ctx_ps1)
        # q_z0_context_prior = distributions.MultivariateNormalDiag(np.zeros_like(mu_z0_ctx_ps1), np.ones_like(sigma_z0_ctx_ps1))
        q_z0_tgt = distributions.MultivariateNormalDiag(mu_z0_tgt, sigma_z0_tgt)

        # This line of code follows Jiang et al 2022, we may want to change here!
        q_z0_tgt_prior = distributions.MultivariateNormalDiag(
            np.zeros_like(mu_z0_tgt), np.ones_like(sigma_z0_tgt)
        )

        q_g_dynamic_all = distributions.MultivariateNormalDiag(
            mu_sys_dynamic_tgt, sigma_sys_dynamic_tgt
        )  # [batch_size, traj_size, dim]

        q_g_dynamic_prior = distributions.MultivariateNormalDiag(
            np.zeros_like(mu_sys_dynamic_tgt), np.ones_like(sigma_sys_dynamic_tgt)
        )

        # calculate the vi sampled log likelihood on the target data
        # we only use the 1st data as our target data
        expanded_target_x = np.expand_dims(
            data_x, axis=2
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, traj_size, num_samples, num_points, 1]

        masked_MC_avg_target_log_likelihood = MC_avg_target_log_likelihood * np.expand_dims(  # expand with the sample_size dimension as well as the states dimension
            np.expand_dims(target_mask, axis=2), axis=-1
        ).astype(
            data_x.dtype
        )
        # sum log likelihood and average through sample size and MC trajectories
        MC_avg_target_log_likelihood = np.mean(
            np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]

        kl_g_dynamic = np.mean(
            distributions.kl_divergence(q_g_dynamic_all, q_g_dynamic_prior), axis=-1
        )
        kl_z0_tgt = np.mean(
            distributions.kl_divergence(q_z0_tgt, q_z0_tgt_prior), axis=-1
        )

        masked_form_objective = -np.mean(
            MC_avg_target_log_likelihood - (kl_g_dynamic + kl_z0_tgt)
        )

        # note that all the rest are used for supervising the training process
        return masked_form_objective, (
            ode_steps,
            -np.mean(MC_avg_target_log_likelihood),
            0.0,
            np.mean(kl_g_dynamic),
            np.mean(kl_z0_tgt),
        )

    return neural_ode_process_mean_field_varaitional_inference_loss


def GetNeuralODEProcessV2MFVILossCondz0ConddsysLoss(**kwargs):
    """
    Problem setting 1 corresponding loss:
    here we use posterior z0 and posterior dsys
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)

        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt_x0, mask_ctx_x0, mask_ctx_with_new_traj: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            target_initial_cond_mask=mask_tgt_x0,
            ctx_mask_with_new_traj_obs=mask_ctx_x0,
            ctx_mask_with_new_traj_target_mask=mask_ctx_with_new_traj,
            solver="Dopri5",
            t0=t0,
            t1=t1,
        )
        (
            x_pred_f,
            x_pred_sigma,
            mu_z0_tgt,
            sigma_z0_tgt,
            mu_traj_dynamic_ctx,
            sigma_traj_dynamic_ctx,
            mu_traj_dynamic_tgt,
            sigma_traj_dynamic_tgt,
            mu_sys_dynamic_ctx,
            sigma_sys_dynamic_ctx,
            mu_sys_dynamic_tgt,
            sigma_sys_dynamic_tgt,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_initial_cond_mask,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
        )
        # this does not need to modify
        p_x_pred = distributions.Normal(
            x_pred_f, x_pred_sigma
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]

        # q_z0_context = distributions.MultivariateNormalDiag(mu_z0_ctx, sigma_z0_ctx)
        # q_z0_target = distributions.MultivariateNormalDiag(mu_z0_tgt, sigma_z0_tgt)
        q_traj_context = distributions.MultivariateNormalDiag(mu_traj_dynamic_ctx, sigma_traj_dynamic_ctx)
        q_traj_target = distributions.MultivariateNormalDiag(mu_traj_dynamic_tgt, sigma_traj_dynamic_tgt)

        q_g_dynamic_context = distributions.MultivariateNormalDiag(
            mu_sys_dynamic_ctx, sigma_sys_dynamic_ctx
        )  # [batch_size, traj_size, dim]
        q_g_dynamic_all = distributions.MultivariateNormalDiag(
            mu_sys_dynamic_tgt, sigma_sys_dynamic_tgt
        )  # [batch_size, traj_size, dim]

        # calculate the vi sampled log likelihood on the target data
        # we only use the 1st data as our target data
        expanded_target_x = np.expand_dims(
            data_x, axis=2
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, traj_size, num_samples, num_points, 1]

        masked_MC_avg_target_log_likelihood = MC_avg_target_log_likelihood * np.expand_dims(  # expand with the sample_size dimension as well as the states dimension
            np.expand_dims(target_mask, axis=2), axis=-1
        ).astype(
            data_x.dtype
        )
        # sum log likelihood and average through sample size and MC trajectories
        MC_avg_target_log_likelihood = np.mean(
            np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]

        kl_g_dynamic = np.mean(
            distributions.kl_divergence(q_g_dynamic_all, q_g_dynamic_context), axis=-1
        )

        kl_traj_dynamic = np.mean(
            distributions.kl_divergence(q_traj_target, q_traj_context), axis=-1
        )
        masked_form_objective = -np.mean(MC_avg_target_log_likelihood - (kl_g_dynamic + kl_traj_dynamic))

        # note that all the rest are used for supervising the training process
        return masked_form_objective, (
            ode_steps,
            -np.mean(MC_avg_target_log_likelihood),
            0.0,
            np.mean(kl_g_dynamic),
            np.mean(kl_traj_dynamic),
        )

    return neural_ode_process_mean_field_varaitional_inference_loss



def GetBatchNeuralODEProcessMCLogLikelihoodLoss(**kwargs):
    """
    Implementation of the Monte Carlo based log lieklihood maximization

    detailed refer to foong2020meta
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
        data_t, data_x, context_mask, target_mask = batch
        data_t = np.squeeze(data_t, axis=-1)
        sort_indices = np.argsort(
            data_t, axis=-1
        )  # [batch_size, traj_size, num_points]
        data_t = np.take_along_axis(
            data_t, sort_indices, axis=-1
        )  # [batch_size, traj_size, num_points]
        expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=-2)
        context_mask = np.take_along_axis(context_mask, sort_indices, axis=-1)
        target_mask = np.take_along_axis(target_mask, sort_indices, axis=-1)
        batch_model_apply = (
            lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: model.apply(
                params,
                t_context=tctx,
                x_context=x_ctx,
                t_target=t_tgt,
                context_mask=mask_ctx,
                target_mask=mask_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=x_tgt,
                training=training,
                solver="Dopri5",
                t0=t0,
                t1=t1,
            )
        )

        x_pred_f, x_pred_sigma, _, _, _, _, _, _, _, _, _ = jax.vmap(batch_model_apply)(
            data_t, data_x, data_t, data_x, context_mask, target_mask
        )

        # we only use the 1st data as our target data
        only_first_target_mask = target_mask[:, 0, :]  # [batch_size, num_points]
        p_x_pred = distributions.Normal(
            x_pred_f, x_pred_sigma
        )  # [batch_size, sample_size, num_points, output_dim]
        expanded_target_x = np.expand_dims(
            data_x[:, 0, :, :], axis=1
        )  # [batch_size, 1, num_points, output_dim]
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, num_samples, num_points, output_dim]
        masked_MC_avg_target_log_likelihood = (
            MC_avg_target_log_likelihood
            * np.expand_dims(
                np.expand_dims(only_first_target_mask, axis=1), axis=-1
            ).astype(data_x.dtype)
        )
        # MC_avg_target_log_likelihood = np.mean(np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=1) # [batch_size]
        # note the main difference is the average is taken before the logarithm opertaion
        MC_log_avg_target_likelihood = np.log(
            np.maximum(
                np.mean(
                    np.exp(np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2])),
                    axis=1,
                ),
                1e-20,
            )
        )  # [batch_size]

        return -np.mean(MC_log_avg_target_likelihood), np.inf

    return neural_process_mean_field_varaitional_inference_loss


def GetGreyBoxNeuralODEProcessMFVILossConddsysLoss(**kwargs):
    """
    Problem setting 1 corresponding loss:
    here we use posterior z0 and posterior dsys
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1
    _stability_epsilon = 1E-20
    d_penality_weight = kwargs["config"].loss.d_penality_weight if hasattr(kwargs["config"], "loss") and hasattr(kwargs["config"].loss, "d_penality_weight") else 1.0
    likelihood_reg_weight = kwargs["config"].loss.likelihood_reg_weight if hasattr(kwargs["config"], "loss") and hasattr(kwargs["config"].loss, "likelihood_reg_weight") else 0.0

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)

        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt_x0, mask_ctx_x0, mask_ctx_with_new_traj: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            target_initial_cond_mask=mask_tgt_x0,
            ctx_mask_with_new_traj_obs=mask_ctx_x0,
            ctx_mask_with_new_traj_target_mask=mask_ctx_with_new_traj,
            solver="Dopri5",
            t0=t0,
            t1=t1,
        )
        (
            x_pred_f,
            x_pred_sigma,
            mu_params_ctx, 
            sigma_params_ctx, 
            mu_params_tgt,
            sigma_params_tgt,
            penalized_distance,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_initial_cond_mask,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
        )
        # clamp x_pred_f by the threshold
        # x_pred_f = np.clip(x_pred_f, _clamp_lower_threshold, _clamp_upper_threshold)
        # this does not need to modify
        p_x_pred = distributions.Normal(
            x_pred_f, x_pred_sigma
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]

        q_params_context = distributions.LogNormal(
            mu_params_ctx, sigma_params_ctx
        )
        q_params_all = distributions.LogNormal(mu_params_tgt, sigma_params_tgt)
        
        expanded_target_x = np.expand_dims(
            data_x, axis=2
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]

        MC_avg_target_log_likelihood = np.log(np.maximum(p_x_pred.prob(expanded_target_x), _stability_epsilon))  # [batch_size, traj_size, num_samples, num_points, 1]

        masked_MC_avg_target_log_likelihood = MC_avg_target_log_likelihood * np.expand_dims(  # expand with the sample_size dimension as well as the states dimension
            np.expand_dims(target_mask, axis=2), axis=-1
        ).astype(
            data_x.dtype
        )
        # sum log likelihood and average through sample size and MC trajectories
        MC_avg_target_log_likelihood = np.mean(
            np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]

        # kl_g_dynamic = np.mean(
        #     distributions.kl_divergence(q_g_dynamic_all, q_g_dynamic_context), axis=-1
        # )
        kl_params = np.mean(
            np.sum(distributions.kl_divergence(q_params_all, q_params_context), axis=-1), axis=-1
        )
        # likelihood reg term
        log_le_reg = np.mean(likelihood_reg_weight * np.sum(q_params_all.log_prob(data_params), axis=-1), axis=-1)
        # 
        masked_form_objective = -np.mean(MC_avg_target_log_likelihood - (kl_params) - d_penality_weight * penalized_distance + log_le_reg)

        # note that all the rest are used for supervising the training process
        return masked_form_objective, (
            ode_steps,
            -np.mean(MC_avg_target_log_likelihood),
            0.0,
            np.mean((kl_params)),
            0.0,
        )

    return neural_ode_process_mean_field_varaitional_inference_loss


def ExperimentalGetNeuralODEProcessMFVILoss(**kwargs):
    """
    Neural ODE Processes Mean Field Variational Inference losses, the defualt choice used
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
        data_t, data_x, context_mask, target_mask = batch
        data_t = np.squeeze(data_t, axis=-1)

        sort_indices = np.argsort(data_t, axis=1)
        data_t = np.take_along_axis(data_t, sort_indices, axis=1)
        expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=1)
        context_mask = np.take_along_axis(context_mask, sort_indices, axis=1)
        target_mask = np.take_along_axis(target_mask, sort_indices, axis=1)

        batch_model_apply = (
            lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: model.apply(
                params,
                t_context=tctx,
                x_context=x_ctx,
                t_target=t_tgt,
                context_mask=mask_ctx,
                target_mask=mask_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=x_tgt,
                training=training,
                solver="Dopri5",
                t0=t0,
                t1=t1,
            )
        )

        (
            x_pred_f,
            x_pred_sigma,
            mu_z0_ctx,
            sigma_z0_ctx,
            mu_z0_tgt,
            sigma_z0_tgt,
            mu_global_ctx,
            sigma_global_ctx,
            mu_global_tgt,
            sigma_global_tgt,
        ) = jax.vmap(batch_model_apply)(
            data_t, data_x, data_t, data_x, context_mask, target_mask
        )
        p_x_pred = distributions.Normal(x_pred_f, x_pred_sigma)
        q_z0_context = distributions.MultivariateNormalDiag(mu_z0_ctx, sigma_z0_ctx)
        q_z0_all = distributions.MultivariateNormalDiag(mu_z0_tgt, sigma_z0_tgt)

        q_global_context = distributions.MultivariateNormalDiag(
            mu_global_ctx, sigma_global_ctx
        )
        q_global_all = distributions.MultivariateNormalDiag(
            mu_global_tgt, sigma_global_tgt
        )
        expanded_target_x = np.expand_dims(data_x, axis=1)
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, num_samples, num_points, 1]
        masked_MC_avg_target_log_likelihood = (
            MC_avg_target_log_likelihood
            * np.expand_dims(np.expand_dims(target_mask, axis=1), axis=-1)
        )
        MC_avg_target_log_likelihood = np.squeeze(
            np.mean(np.sum(masked_MC_avg_target_log_likelihood, axis=-2), axis=1),
            axis=-1,
        )  # [batch_size, 1]

        kl_z0 = distributions.kl_divergence(q_z0_all, q_z0_context)
        kl_control = distributions.kl_divergence(q_global_all, q_global_context)
        masked_form_objective = -np.mean(
            MC_avg_target_log_likelihood - kl_z0 - kl_control
        )
        return masked_form_objective

    return neural_ode_process_mean_field_varaitional_inference_loss


def GetNeuralODEProcessTaylorLikelihoodLoss(**kwargs):
    """
    Neural Processes Mean Field Variational Inference losses, the default choice used
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_process_taylor_likelihood_loss(rng, params, batch):
        
        data_t, data_x, context_mask, target_mask = batch
        data_t = np.squeeze(data_t, axis=-1)

        sort_indices = np.argsort(data_t, axis=1)
        data_t = np.take_along_axis(data_t, sort_indices, axis=1)
        expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=1)
        context_mask = np.take_along_axis(context_mask, sort_indices, axis=1)
        target_mask = np.take_along_axis(target_mask, sort_indices, axis=1)
        batch_model_apply = (
            lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: model.apply(
                params,
                t_context=tctx,
                x_context=x_ctx,
                t_target=t_tgt,
                context_mask=mask_ctx,
                target_mask=mask_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=x_tgt,
                training=training,
                solver="Dopri5",
                t0=t0,
                t1=t1,
            )
        )

        (
            x_pred_f,
            x_pred_sigma,
            mu_z0_ctx,
            sigma_z0_ctx,
            mu_z0_tgt,
            sigma_z0_tgt,
            mu_global_ctx,
            sigma_global_ctx,
            mu_global_tgt,
            sigma_global_tgt,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t, data_x, data_t, data_x, context_mask, target_mask
        )
        expanded_target_x = np.expand_dims(data_x, axis=1)
        MC_avg_target_log_likelihood = distributions.Normal(
            x_pred_f, x_pred_sigma
        ).log_prob(expanded_target_x) * np.expand_dims(
            np.expand_dims(target_mask, axis=1), axis=-1
        )  # [batch_size, num_samples, num_points, output_dim]
        MC_avg_target_log_likelihood = np.mean(
            np.sum(MC_avg_target_log_likelihood, axis=[-1, -2]), axis=1
        )  # [batch_size, 1]

        return -np.mean(MC_avg_target_log_likelihood), np.inf

    return neural_process_taylor_likelihood_loss


def GetNeuralODEProcessTestMSELoss(**kwargs):
    """
    Neural ODE Processes Mean Field Variational Inference losses, the defualt choice used
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mse_loss(rng, params, batch):
        
        data_t, data_x, context_mask, target_mask = batch
        # 2/14/2024 临时加的
        data_t, data_x, context_mask, target_mask = (
            data_t[:, 0, ...],
            data_x[:, 0, ...],
            context_mask[:, 0, ...],
            target_mask[:, 0, ...],
        )
        data_t = np.squeeze(data_t, axis=-1)

        sort_indices = np.argsort(data_t, axis=1)
        data_t = np.take_along_axis(data_t, sort_indices, axis=1)
        expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=1)
        context_mask = np.take_along_axis(context_mask, sort_indices, axis=1)
        target_mask = np.take_along_axis(target_mask, sort_indices, axis=1)
        batch_model_apply = (
            lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: model.apply(
                params,
                t_context=tctx,
                x_context=x_ctx,
                t_target=t_tgt,
                context_mask=mask_ctx,
                target_mask=mask_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=x_tgt,
                training=training,
                solver="Dopri5",
                t0=t0,
                t1=t1,
            )
        )

        x_pred_f, _ = jax.vmap(batch_model_apply)(
            data_t, data_x, data_t, data_x, context_mask, target_mask
        )
        expanded_aug_x = np.expand_dims(data_x, axis=1)
        # mse = np.mean(np.sum(((x_pred_f - expanded_aug_x) ** 2) * np.expand_dims(np.expand_dims(target_mask, axis=1), axis=-1), axis=-2))
        mse = np.mean(  # average through batch
            np.sum(
                ((x_pred_f - expanded_aug_x) ** 2)
                * np.expand_dims(np.expand_dims(target_mask, axis=1), axis=-1),
                axis=[-1, -2],
            )
            / np.expand_dims(np.sum(target_mask, axis=-1), axis=-1)
        )

        return mse, np.inf  # np.inf represent ode step

    return neural_ode_process_mse_loss


def GetNeuralODEProcessTestMSELossAcceptBatchData(**kwargs):
    """
    Neural ODE Processes Mean Field Variational Inference losses, the defualt choice used
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mse_loss(rng, params, batch):
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)
        context_mask  = np.diagonal(ctx_mask_with_new_traj_obs, axis1=-3, axis2=-2)
        target_mask = np.diagonal(ctx_mask_with_new_traj_target_mask, axis1=-3, axis2=-2)
        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt, mask_ctx: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            context_mask = mask_ctx, 
            target_mask = mask_tgt,
            solver="Dopri5",
            t0=t0,
            t1=t1)
        (
            x_pred_f,
            x_pred_sigma
        ) = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_mask,
            context_mask,
        )
        
        # use the test aligned version
        target_mask = np.diagonal(ctx_mask_with_new_traj_target_mask, axis1=-3, axis2=-2)
        element_wise_mse = ((x_pred_f - np.expand_dims(data_x, axis=2)) ** 2) * np.expand_dims(np.expand_dims(target_mask, axis=2), axis=-1)
        ml_mse = np.sum(np.mean(element_wise_mse, axis=-1),axis=-1) / np.maximum(np.expand_dims(np.sum(target_mask, axis=-1), axis=-1), 1.0)

        ml_mse_traj_mc_wise_mean = np.mean(ml_mse, axis=[-1, -2])
        return np.mean(ml_mse_traj_mc_wise_mean), (0.0, -np.mean(ml_mse_traj_mc_wise_mean), 0.0, 0.0, 0.0)

        # mse = np.mean(
        #     np.sum(
        #         np.mean(((x_pred_f - np.expand_dims(data_x, axis=2)) ** 2)
        #         * np.expand_dims(np.expand_dims(target_mask, axis=2), axis=-1), axis=-1),
        #         axis=-1
        #     )
        #     / np.maximum(np.expand_dims(np.sum(target_mask, axis=-1), axis=-1), 1.0),
        #     axis=[-1, -2],
        # )
        # return np.mean(mse), (0.0, -np.mean(mse), 0.0, 0.0, 0.0)

    return neural_ode_process_mse_loss


def GetNeuralODEProcessNLLLoss(**kwargs):
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_nll_loss(rng, params, batch):
        
        data_t, data_x, context_mask, target_mask = batch
        # 2/14/2024 临时加的
        data_t, data_x, context_mask, target_mask = (
            data_t[:, 0, ...],
            data_x[:, 0, ...],
            context_mask[:, 0, ...],
            target_mask[:, 0, ...],
        )
        data_t = np.squeeze(data_t, axis=-1)

        sort_indices = np.argsort(data_t, axis=1)
        data_t = np.take_along_axis(data_t, sort_indices, axis=1)
        expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=1)
        context_mask = np.take_along_axis(context_mask, sort_indices, axis=1)
        target_mask = np.take_along_axis(target_mask, sort_indices, axis=1)
        batch_model_apply = (
            lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: model.apply(
                params,
                t_context=tctx,
                x_context=x_ctx,
                t_target=t_tgt,
                context_mask=mask_ctx,
                target_mask=mask_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=x_tgt,
                training=training,
                solver="Dopri5",
                t0=t0,
                t1=t1,
            )
        )

        x_pred_f, x_pred_sigma = jax.vmap(batch_model_apply)(
            data_t, data_x, data_t, data_x, context_mask, target_mask
        )
        expanded_aug_x = np.expand_dims(data_x, axis=1)
        # MC_avg_target_log_likelihood = distributions.Normal(
        #     x_pred_f, x_pred_sigma
        # ).log_prob(expanded_aug_x) * np.expand_dims(
        #     np.expand_dims(target_mask, axis=1), axis=-1
        # )  # [batch_size, num_samples, num_points, 1]
        # MC_avg_target_log_likelihood = np.mean(
        #     np.sum(MC_avg_target_log_likelihood, axis=[-1, -2]), axis=1
        # )  # [batch_size]

        # use the test aligned version
        MC_avg_target_log_likelihood = distributions.Normal(x_pred_f, x_pred_sigma).log_prob(expanded_aug_x) # (20, 100, 32, 100, 2)
        # [batch_size, traj_size, num_samples, num_points, 1]
        # target_mask: [batch_size, traj_size, num_points]
        MC_avg_target_log_likelihood = MC_avg_target_log_likelihood * \
            np.expand_dims(np.expand_dims(target_mask, axis=2), axis=-1).astype(data_x.dtype)
        MC_avg_target_log_likelihood = np.mean(
            np.sum(MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]
        return (
            -np.mean(MC_avg_target_log_likelihood),
            np.inf,
        )  # np.inf represent ode step

    return neural_ode_process_nll_loss


def GetVectorFieldAttenNeuralODEProcessMFVILossLoss(**kwargs):
    """
    Problem setting 1 corresponding loss:
    here we use posterior z0 and posterior dsys
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_mean_field_varaitional_inference_loss(
        rng, params, batch
    ):
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)

        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt_x0, mask_ctx_x0, mask_ctx_with_new_traj: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            target_initial_cond_mask=mask_tgt_x0,
            ctx_mask_with_new_traj_obs=mask_ctx_x0,
            ctx_mask_with_new_traj_target_mask=mask_ctx_with_new_traj,
            solver="Dopri5",
            t0=t0,
            t1=t1,
        )
        (
            x_pred_f,
            x_pred_sigma,
            mu_inducing_vec_ctx,
            sigma_inducing_vec_ctx,
            mu_inducing_vec_tgt,
            sigma_inducing_vec_tgt,
            ode_steps,
        ) = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_initial_cond_mask,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
        )
        # this does not need to modify
        p_x_pred = distributions.Normal(
            x_pred_f, x_pred_sigma
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]

        q_inducing_vec_ctx = distributions.MultivariateNormalDiag(
            mu_inducing_vec_ctx, sigma_inducing_vec_ctx
        )  # [batch_size, traj_size, N_inducing, dim]
        q_inducing_vec_tgt = distributions.MultivariateNormalDiag(
            mu_inducing_vec_tgt, sigma_inducing_vec_tgt
        )  # [batch_size, traj_size, N_inducing, dim]

        # calculate the vi sampled log likelihood on the target data
        # we only use the 1st data as our target data
        expanded_target_x = np.expand_dims(
            data_x, axis=2
        )  # [batch_size, traj_size, sample_size, num_points, output_dim]
        MC_avg_target_log_likelihood = p_x_pred.log_prob(
            expanded_target_x
        )  # [batch_size, traj_size, num_samples, num_points, 1]

        # MC_avg_target_log_likelihood = np.where(
        #     np.isnan(MC_avg_target_log_likelihood),
        #     -1000000000,
        #     MC_avg_target_log_likelihood
        # )

        masked_MC_avg_target_log_likelihood = MC_avg_target_log_likelihood * np.expand_dims(  # expand with the sample_size dimension as well as the states dimension
            np.expand_dims(target_mask, axis=2), axis=-1
        ).astype(
            data_x.dtype
        )
        # sum log likelihood and average through sample size and MC trajectories
        MC_avg_target_log_likelihood = np.mean(
            np.sum(masked_MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]

        kl_g_dynamic = np.mean(
            distributions.kl_divergence(q_inducing_vec_tgt, q_inducing_vec_ctx), axis=[-1, -2]
        )

        masked_form_objective = -np.mean(MC_avg_target_log_likelihood - (kl_g_dynamic))

        # note that all the rest are used for supervising the training process
        return masked_form_objective, (
            ode_steps,
            -np.mean(MC_avg_target_log_likelihood),
            0.0,
            np.mean(kl_g_dynamic),
            0.0,
        )

    return neural_ode_process_mean_field_varaitional_inference_loss


def GetNeuralODEProcessNLLLossAcceptBatchData(**kwargs):
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_nll_loss(rng, params, batch):
        
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
        ) = batch
        # we do not use any sorting here since it does not make too much sense
        data_t = np.squeeze(data_t, axis=-1)
        context_mask  = np.diagonal(ctx_mask_with_new_traj_obs, axis1=-3, axis2=-2)
        target_mask = np.diagonal(ctx_mask_with_new_traj_target_mask, axis1=-3, axis2=-2)
        batch_model_apply = lambda tctx, x_ctx, t_tgt, x_tgt, mask_tgt, mask_ctx: model.apply(
            params,
            t_context=tctx,
            x_context=x_ctx,
            t_target=t_tgt,
            sample_rng=rng,
            sample_size=sample_size,
            x_target=x_tgt,
            training=training,
            context_mask = mask_ctx, 
            target_mask = mask_tgt,
            solver="Dopri5",
            t0=t0,
            t1=t1)
        (
            x_pred_f,
            x_pred_sigma
        ) = jax.vmap(batch_model_apply)(
            data_t,
            data_x,
            data_t,
            data_x,
            target_mask,
            context_mask,
        )
        expanded_aug_x = np.expand_dims(data_x, axis=2)
        MC_avg_target_log_likelihood = distributions.Normal(x_pred_f, x_pred_sigma).log_prob(expanded_aug_x) # (20, 100, 32, 100, 2)
        # [batch_size, traj_size, num_samples, num_points, 1]
        # target_mask: [batch_size, traj_size, num_points]
        MC_avg_target_log_likelihood = MC_avg_target_log_likelihood * \
            np.expand_dims(np.expand_dims(target_mask, axis=2), axis=-1).astype(data_x.dtype)
        MC_avg_target_log_likelihood = np.mean(
            np.sum(MC_avg_target_log_likelihood, axis=[-1, -2]), axis=[-1, -2]
        )  # [batch_size]
        return (
            -np.mean(MC_avg_target_log_likelihood), 
            (0.0, -np.mean(MC_avg_target_log_likelihood), 0.0, 0.0, 0.0)
        )  # np.inf represent ode step

    return neural_ode_process_nll_loss


def GetNeuralODEProcessEnsembleCELoss(**kwargs):
    """
    Implementation of the ensemble calibration error
    which is documented in C.12-C.17 of bootstraping neural process paper
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    levels = kwargs["config"].data.aux_cfg["CE"]["levels"] = 10
    level_ind = np.linspace(0.0, 1.0, levels + 1, endpoint=False)[1:]  # [levels]
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_ensemble_ce_loss(rng, params, batch):
        
        data_t, data_x, context_mask, target_mask = batch
        data_t = np.squeeze(data_t, axis=-1)

        sort_indices = np.argsort(data_t, axis=1)
        data_t = np.take_along_axis(data_t, sort_indices, axis=1)
        expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=1)
        context_mask = np.take_along_axis(context_mask, sort_indices, axis=1)
        target_mask = np.take_along_axis(target_mask, sort_indices, axis=1)
        batch_model_apply = (
            lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: model.apply(
                params,
                t_context=tctx,
                x_context=x_ctx,
                t_target=t_tgt,
                context_mask=mask_ctx,
                target_mask=mask_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=x_tgt,
                training=training,
                solver="Dopri5",
                t0=t0,
                t1=t1,
            )
        )

        x_pred_f, x_pred_sigma = jax.vmap(batch_model_apply)(
            data_t, data_x, data_t, data_x, context_mask, target_mask
        )
        res = ppf(
            loc=x_pred_f[..., None], scale=x_pred_sigma[..., None], q=level_ind
        )  # [batch_size, num_samples, num_points, output_dim, levels]
        aug_data_y = np.expand_dims(
            np.expand_dims(data_x, axis=1), axis=-1
        )  # [batch_size, num_samples, num_points, output_dim, levels]
        expand_target_mask = np.expand_dims(target_mask, axis=1)[..., None, None]
        p_hat = np.sum((res > aug_data_y) * expand_target_mask, axis=2) / np.sum(
            expand_target_mask, axis=-3
        )  # [batch_size, num_samples, output_dim, levels]

        p_diff = (
            p_hat - level_ind
        ) ** 2  # [batch_size, num_samples, output_dim, levels]
        p_CE = np.mean(np.sum(p_diff, axis=-1), axis=[1, -1])  # [batch_size]
        return np.mean(p_CE), np.inf  # hack for steps

    return neural_ode_process_ensemble_ce_loss


def GetNeuralODEProcessSharpnessLoss(**kwargs):
    """
    Implementation of the sharpness loss
    which is documented in C.18 of bootstraping neural process paper
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_sharpness_loss(rng, params, batch):
        
        data_t, data_x, context_mask, target_mask = batch
        data_t = np.squeeze(data_t, axis=-1)

        sort_indices = np.argsort(data_t, axis=1)
        data_t = np.take_along_axis(data_t, sort_indices, axis=1)
        expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=1)
        context_mask = np.take_along_axis(context_mask, sort_indices, axis=1)
        target_mask = np.take_along_axis(target_mask, sort_indices, axis=1)
        batch_model_apply = (
            lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: model.apply(
                params,
                t_context=tctx,
                x_context=x_ctx,
                t_target=t_tgt,
                context_mask=mask_ctx,
                target_mask=mask_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=x_tgt,
                training=training,
                solver="Dopri5",
                t0=t0,
                t1=t1,
            )
        )

        x_pred_sigma = jax.vmap(batch_model_apply)(
            data_t, data_x, data_t, data_x, context_mask, target_mask
        )
        expanded_target_mask = np.expand_dims(
            np.expand_dims(target_mask, axis=1), axis=-1
        )
        screened_var = (
            x_pred_sigma * expanded_target_mask
        ) ** 2  # [batch_size, num_samples, num_points, 1]
        averaged_sharpness = np.mean(
            np.sum(screened_var, axis=-2) / np.sum(expanded_target_mask, -2),
            axis=[1, 2],
        )  # [batch_size]
        return np.mean(averaged_sharpness), np.inf  # hack for steps

    return neural_ode_process_sharpness_loss


def GetNeuralODEProcessEnsembleCELoss(**kwargs):
    """
    Implementation of the ensemble calibration error
    which is documented in C.12-C.17 of bootstraping neural process paper
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    levels = kwargs["config"].data.aux_cfg["CE"]["levels"] = 10
    level_ind = np.linspace(0.0, 1.0, levels + 1, endpoint=False)[1:]  # [levels]
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_ensemble_ce_loss(rng, params, batch):
        
        data_t, data_x, context_mask, target_mask = batch
        data_t = np.squeeze(data_t, axis=-1)

        sort_indices = np.argsort(data_t, axis=1)
        data_t = np.take_along_axis(data_t, sort_indices, axis=1)
        expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=1)
        context_mask = np.take_along_axis(context_mask, sort_indices, axis=1)
        target_mask = np.take_along_axis(target_mask, sort_indices, axis=1)
        batch_model_apply = (
            lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: model.apply(
                params,
                t_context=tctx,
                x_context=x_ctx,
                t_target=t_tgt,
                context_mask=mask_ctx,
                target_mask=mask_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=x_tgt,
                training=training,
                solver="Dopri5",
                t0=t0,
                t1=t1,
            )
        )

        x_pred_f, x_pred_sigma = jax.vmap(batch_model_apply)(
            data_t, data_x, data_t, data_x, context_mask, target_mask
        )
        res = ppf(
            loc=x_pred_f[..., None], scale=x_pred_sigma[..., None], q=level_ind
        )  # [batch_size, num_samples, num_points, output_dim, levels]
        aug_data_x = np.expand_dims(
            np.expand_dims(data_x, axis=1), axis=-1
        )  # [batch_size, num_samples, num_points, output_dim, levels]
        expand_target_mask = np.expand_dims(target_mask, axis=1)[..., None, None]
        p_hat = np.sum((res > aug_data_x) * expand_target_mask, axis=2) / np.sum(
            expand_target_mask, axis=-3
        )  # [batch_size, num_samples, output_dim, levels]

        p_diff = (
            p_hat - level_ind
        ) ** 2  # [batch_size, num_samples, output_dim, levels]
        p_CE = np.mean(np.sum(p_diff, axis=-1), axis=[1, -1])  # [batch_size]
        return np.mean(p_CE), np.inf  # hack for steps

    return neural_ode_process_ensemble_ce_loss


def GetNeuralODEProcessSharpnessLoss(**kwargs):
    """
    Implementation of the sharpness loss
    which is documented in C.18 of bootstraping neural process paper
    """
    model, training = kwargs["model"], kwargs["training"]
    sample_size = kwargs["config"].model.sample_size
    t0 = kwargs["config"].model.t0
    t1 = kwargs["config"].model.t1

    @jax.jit
    def neural_ode_process_sharpness_loss(rng, params, batch):
        
        data_t, data_x, context_mask, target_mask = batch
        data_t = np.squeeze(data_t, axis=-1)

        sort_indices = np.argsort(data_t, axis=1)
        data_t = np.take_along_axis(data_t, sort_indices, axis=1)
        expanded_sort_indices = np.expand_dims(sort_indices, axis=-1)
        data_x = np.take_along_axis(data_x, expanded_sort_indices, axis=1)
        context_mask = np.take_along_axis(context_mask, sort_indices, axis=1)
        target_mask = np.take_along_axis(target_mask, sort_indices, axis=1)
        batch_model_apply = (
            lambda tctx, x_ctx, t_tgt, x_tgt, mask_ctx, mask_tgt: model.apply(
                params,
                t_context=tctx,
                x_context=x_ctx,
                t_target=t_tgt,
                context_mask=mask_ctx,
                target_mask=mask_tgt,
                sample_rng=rng,
                sample_size=sample_size,
                x_target=x_tgt,
                training=training,
                solver="Dopri5",
                t0=t0,
                t1=t1,
            )
        )
        x_pred_f, x_pred_sigma = jax.vmap(batch_model_apply)(
            data_t, data_x, data_t, data_x, context_mask, target_mask
        )
        expanded_target_mask = np.expand_dims(
            np.expand_dims(target_mask, axis=1), axis=-1
        )
        screened_var = (
            x_pred_sigma * expanded_target_mask
        ) ** 2  # [batch_size, num_samples, num_points, 1]
        averaged_sharpness = np.mean(
            np.sum(screened_var, axis=-2) / np.sum(expanded_target_mask, -2),
            axis=[1, 2],
        )  # [batch_size]
        return np.mean(averaged_sharpness), np.inf  # hack for steps

    return neural_ode_process_sharpness_loss


def get_train_eval_step_fn(
    model, training: bool, optimize_fn: Optional[Callable], loss_method: str, config
):
    """Create a one-step training/evaluation function.

    Args:
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      training: `True` for training and `False` for evaluation.
      optimize_fn: An optimization function that will return optimized new parameter
      loss_method: the name of the loss function.

    Returns:
      A one-step function for training or evaluation.
    """
    loss_fn = getattr(sys.modules[__name__], "Get" + loss_method)(
        model=model, training=training, config=config
    )

    @jax.jit
    def step_fn(carry_state: Tuple[TrainState], batch: jax.Array):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
          batch: A mini-batch of training/evaluation data.

        Returns:
          new_carry_state: The updated tuple of `carry_state`.
          loss: The average loss value of this state.
        """

        (rng, train_state) = carry_state
        rng, step_rng = jax.random.split(rng)
        params = train_state.params
        if training:
            assert optimize_fn is not None

            grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
            (loss, (ode_steps, NLL, kl_control, kl_g_dynamic, kl_z0_tgt)), grad = (
                grad_fn(step_rng, params, batch)
            )
            
            new_train_state = train_state.apply_gradients(grads=grad)
            aux_info = {
                "ode_steps": ode_steps,
                "nll": NLL,
                "kl_ctrl": kl_control,
                "kl_dnmc": kl_g_dynamic,
                "kl_z0": kl_z0_tgt,
            }
        else:
            NLL, kl_control, kl_g_dynamic, kl_z0_tgt = None, None, None, None
            (loss, ode_steps) = loss_fn(step_rng, params, batch)
            new_train_state = train_state
            aux_info = {
                "ode_steps": ode_steps,
                "nll": NLL,
                "kl_ctrl": kl_control,
                "kl_dnmc": kl_g_dynamic,
                "kl_z0": kl_z0_tgt,
            }
            # aux_info = {
            #     "eval_ode_steps": ode_steps,
            #     "eval_nll": NLL,
            #     "eval_kl_control": kl_control,
            #     "eval_kl_g_dynamic": kl_g_dynamic,
            #     "eval_kl_z0_tgt": kl_z0_tgt,
            # }

        new_carry_state = (rng, new_train_state)

        return (
            new_carry_state,
            loss,
            aux_info
        )

    return step_fn
