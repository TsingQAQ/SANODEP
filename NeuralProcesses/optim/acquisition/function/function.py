"""
Single Objective Acquisition Functions
"""

from trieste.acquisition.function.function import (
    HasReparamSampler,
    Optional,
    Dataset,
    AcquisitionFunction,
    AcquisitionFunctionClass,
    TensorType,
    cast,
    DEFAULTS,
)
from ..interface import CustomizedSingleModelAcquisitionBuilder

import jax
from jax import random
import tensorflow as tf
from jax import numpy as np
from NeuralProcesses.data.datasets import NP_Dataset
from abc import ABC
from jax.typing import ArrayLike


class FlaxModelBasedAcqusitionFunction(ABC):
    """
    flax model based acquisition function, this needs some specific care because
    the original gradient from tensorflow does not work here to optimize the acquisition function
    """


class GreyBoxBatchMonteCarloExpectedImprovement(
    CustomizedSingleModelAcquisitionBuilder
):
    """
    Expected improvement for batches of points (or :math:`q`-EI), approximated using Monte Carlo
    estimation with the reparametrization trick. See :cite:`Ginsbourger2010` for details.
    Improvement is measured with respect to the MAXIMUM predictive mean at observed query points.
    This is calculated in :class:`BatchMonteCarloExpectedImprovement` by assuming observations
    at new points are independent from those at known query points. This is faster, but is an
    approximation for noisy observers.
    """

    def __init__(
        self,
        sample_size: int,
        *,
        obj_func_form: Optional[None] = None,
        jitter: float = DEFAULTS.JITTER,
    ):
        """
        :param sample_size: The number of samples for each batch of points.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :param model_type: The type of the model. either "gpflow" or "flax"
        :raise tf.errors.InvalidArgumentError: If ``sample_size`` is not positive, or ``jitter``
            is negative.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        self._sample_size = sample_size
        self._jitter = jitter
        if obj_func_form is None:
            self.obj_func_form = lambda x: x
        else:
            self.obj_func_form = obj_func_form

    def __repr__(self) -> str:
        """"""
        return f"BatchMonteCarloExpectedImprovement({self._sample_size!r}, jitter={self._jitter!r})"

    def prepare_acquisition_function(
        self, model: HasReparamSampler, dataset: Optional[Dataset] = None, **kwargs
    ) -> AcquisitionFunction:
        """
        :param model: The model. Must have event shape [1].
        :param dataset: The data from the observer. Must be populated.
        :return: The batch *expected improvement* acquisition function.
        :raise ValueError (or InvalidArgumentError): If ``dataset`` is not populated, or ``model``
            does not have an event shape of [1].
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        mean, _ = model.predict(dataset.query_points)
        aggregate_obj_mean = self.obj_func_form(mean)
        tf.debugging.assert_shapes(
            [(aggregate_obj_mean, ["_", 1])],
            message="Expected model with event shape [1].",
        )

        eta = tf.reduce_max(aggregate_obj_mean, axis=0)
        return batch_monte_carlo_expected_improvement(
            self._sample_size,
            model,
            eta,
            self._jitter,
            obj_func_form=self.obj_func_form,
        )


class batch_monte_carlo_expected_improvement(AcquisitionFunctionClass):
    def __init__(
        self,
        sample_size: int,
        model: HasReparamSampler,
        eta: TensorType,
        jitter: float,
        obj_func_form: Optional[None] = None,
    ):
        """
        :param sample_size: The number of Monte-Carlo samples.
        :param model: The model of the objective function.
        :param eta: The "best" observation.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :return: The expected improvement function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        """
        self._sample_size = sample_size

        if not isinstance(model, HasReparamSampler):
            raise ValueError(
                f"The batch Monte-Carlo expected improvement acquisition function only supports "
                f"models that implement a reparam_sampler method; received {model!r}"
            )

        sampler = model.reparam_sampler(self._sample_size)

        self._sampler = sampler
        self._eta = tf.Variable(eta)
        self._jitter = jitter
        self.obj_func_form = obj_func_form

    def update(self, eta: TensorType) -> None:
        """Update the acquisition function with a new eta value and reset the reparam sampler."""
        self._eta.assign(eta)
        self._sampler.reset_sampler()

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        samples = self._sampler.sample(x, jitter=self._jitter)  # [..., S, B, obj_num]
        # min_sample_per_batch = tf.reduce_min(samples, axis=-1)  # [..., S]
        aggregated_samples = tf.squeeze(
            self.obj_func_form(samples), axis=-1
        )  # [..., S, B]
        max_sample_per_batch = tf.reduce_max(aggregated_samples, axis=-1)  # [..., S]
        batch_improvement = tf.maximum(
            max_sample_per_batch - self._eta, 0.0
        )  # [..., S]
        return tf.reduce_mean(batch_improvement, axis=-1, keepdims=True)  # [..., 1]


class GreyBoxBatchMonteCarloExpectedImprovementCompatibleWithFlaxModels(
    GreyBoxBatchMonteCarloExpectedImprovement, FlaxModelBasedAcqusitionFunction
):
    def __init__(
        self,
        sample_size: int,
        sample_rng: random.PRNGKey,
        *,
        obj_func_form: Optional[None] = None,
        jitter: float = DEFAULTS.JITTER,
        trajectory_aware: bool = False,
    ):
        super().__init__(
            sample_size,
            obj_func_form=obj_func_form,
            jitter=jitter,
        )
        self._sample_rng = sample_rng
        self.trajectory_aware = trajectory_aware

    def prepare_acquisition_function(
        self, model: HasReparamSampler, dataset: Optional[NP_Dataset] = None, **kwargs
    ) -> AcquisitionFunction:
        """
        :param model: The model. Must have event shape [1].
        :param dataset: The data from the observer. Must be populated.
        :return: The batch *expected improvement* acquisition function.
        :raise ValueError (or InvalidArgumentError): If ``dataset`` is not populated, or ``model``
            does not have an event shape of [1].
        """
        # tf.debugging.Assert(dataset is not None, [tf.constant([])])
        # dataset = cast(Dataset, dataset)
        # tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        if False:  # self.trajectory_aware:
            times, states, masks = dataset.formalize_training_data_sanodep()
            mean = model.predict(dataset.query_points)
        else:
            dataset = dataset.formalize_training_data_for_trieste(tf.float64)
            mean, _ = model.predict(np.asarray(dataset.query_points.numpy()))
        aggregate_obj_mean = self.obj_func_form(mean)
        tf.debugging.assert_shapes(
            [(aggregate_obj_mean, [..., 1])],
            message="Expected model with event shape [1].",
        )
        eta = tf.reduce_max(aggregate_obj_mean)
        # eta = tf.reduce_max(aggregate_obj_mean, axis=0)
        return batch_monte_carlo_expected_improvement_compatible_with_flax_models(
            model, self._sample_size, self._sample_rng, np.asarray(eta.numpy()), obj_func_form=self.obj_func_form
        ).get_jittable_acq_func()

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: HasReparamSampler,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        pass


class batch_monte_carlo_expected_improvement_compatible_with_flax_models(
    AcquisitionFunctionClass
):
    """
    The only difference of this function compared with batch_monte_carlo_expected_improvement is that we do not wrap
    __call__ with tf.function, this may lead some speed decrease but make it possible to use with flax models
    """

    def __init__(
        self,
        model: HasReparamSampler,
        sample_size: int, 
        sample_rng: random.PRNGKey,
        eta: TensorType,
        obj_func_form: Optional[None] = None,
    ):
        """
        :param sample_size: The number of Monte-Carlo samples.
        :param model: The model of the objective function.
        :param eta: The "best" observation.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :return: The expected improvement function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        """

        sampler = lambda xs, sz, rng: model.sample(xs, sz, rng)  # .reparam_sampler(self._sample_size)

        self._sampler = sampler
        self._eta = eta
        self._sample_size = sample_size
        self._sample_rng = sample_rng
        self.obj_func_form = obj_func_form

    def update(self, eta: TensorType) -> None:
        """Update the acquisition function with a new eta value and reset the reparam sampler."""
        self._eta.assign(eta)
        self._sampler.reset_sampler()

    def get_jittable_acq_func(self):
        @jax.jit
        def single_eval(_x):
            # init_cond, times = _x[..., :0], _x[..., 1:] # [batch_size, state_dim], [batch_size, 1]
            # init_cond = np.stack([2 * init_cond, init_cond], axis=-1)
            # samples = self._sampler(np.concatenate([init_cond, times], axis=-1)) # [timesteps, state]
            samples = self._sampler(_x, self._sample_size, self._sample_rng)  # [timesteps, state]
            # min_sample_per_batch = tf.reduce_min(samples, axis=-1)  # [..., S]
            aggregated_samples = np.squeeze(
                self.obj_func_form(samples), axis=-1
            )  # [..., B, S]
            max_sample_per_batch = np.max(aggregated_samples, axis=-1)
            # max_sample_per_batch = np.max(aggregated_samples, axis=0)  # [..., B, S]
            batch_improvement = np.maximum(
                max_sample_per_batch - self._eta, 0.0
            )  # [..., S]
            # return batch_improvement
            # return np.mean(batch_improvement, axis=-1, keepdims=True)
            return np.mean(batch_improvement, axis=-1, keepdims=True)

        return single_eval

    def __call__(self, x: ArrayLike) -> ArrayLike:
        pass
