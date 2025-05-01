from trieste.models.gpflow.sampler import *
from typing import Optional
from trieste.data import Dataset
import tensorflow as tf

class MultiOutputBatchReparametrizationSampler(BatchReparametrizationSampler):
    def sample(self, at: TensorType, *, jitter: float = DEFAULTS.JITTER) -> TensorType:
        at = tf.pad(at, [[0, 0]] * (len(at.shape) - 1) + [[0, 1]], constant_values=0)
        return super(MultiOutputBatchReparametrizationSampler, self).sample(at, jitter=jitter)
    

class ConditionalBatchReparametrizationSampler(BatchReparametrizationSampler):
    def conditional_sample(self, at: TensorType, *, conditions: Optional[Dataset], jitter: float = DEFAULTS.JITTER) -> TensorType:
        """
        Return approximate samples from the `model` specified at :meth:`__init__`. Multiple calls to
        :meth:`sample`, for any given :class:`BatchReparametrizationSampler` and ``at``, will
        produce the exact same samples. Calls to :meth:`sample` on *different*
        :class:`BatchReparametrizationSampler` instances will produce different samples.

        :param at: Batches of query points at which to sample the predictive distribution, with
            shape `[..., B, D]`, for batches of size `B` of points of dimension `D`. Must have a
            consistent batch size across all calls to :meth:`sample` for any given
            :class:`BatchReparametrizationSampler`.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :return: The samples, of shape `[..., S, B, L]`, where `S` is the `sample_size`, `B` the
            number of points per batch, and `L` the dimension of the model's predictive
            distribution.
        :raise ValueError (or InvalidArgumentError): If any of the following are true:
            - ``at`` is a scalar.
            - The batch size `B` of ``at`` is not positive.
            - The batch size `B` of ``at`` differs from that of previous calls.
            - ``jitter`` is negative.
        """
        tf.debugging.assert_rank_at_least(at, 2)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        batch_size = at.shape[-2]

        tf.debugging.assert_positive(batch_size)
        try:
            mean, cov = self._model.conditional_predict_joint(at, conditions)  # [..., B, L], [..., L, B, B]
        except: # error catch 
            mean, cov = self._model.conditional_predict_joint(at, conditions) 
            
        def sample_eps() -> tf.Tensor:
            self._initialized.assign(True)
            if self._qmc:
                if self._qmc_skip:
                    skip = IndependentReparametrizationSampler.skip
                    IndependentReparametrizationSampler.skip.assign(skip + self._sample_size)
                else:
                    skip = tf.constant(0)
                normal_samples = qmc_normal_samples(
                    self._sample_size * mean.shape[-1], batch_size, skip
                )  # [S*L, B]
                normal_samples = tf.reshape(
                    normal_samples, (mean.shape[-1], self._sample_size, batch_size)
                )  # [L, S, B]
                normal_samples = tf.transpose(normal_samples, perm=[0, 2, 1])  # [L, B, S]
            else:
                normal_samples = tf.random.normal(
                    [tf.shape(mean)[-1], batch_size, self._sample_size], dtype=tf.float64
                )  # [L, B, S]
            return normal_samples

        if self._eps is None:
            # dynamically shaped as the same sampler may be called with different sized batches
            self._eps = tf.Variable(sample_eps(), shape=[None, None, self._sample_size])

        tf.cond(
            self._initialized,
            lambda: self._eps,
            lambda: self._eps.assign(sample_eps()),
        )

        if self._initialized:
            tf.debugging.assert_equal(
                batch_size,
                tf.shape(self._eps)[-2],
                f"{type(self).__name__} requires a fixed batch size. Got batch size {batch_size}"
                f" but previous batch size was {tf.shape(self._eps)[-2]}.",
            )

        identity = tf.eye(batch_size, dtype=cov.dtype)  # [B, B]
        cov_cholesky = tf.linalg.cholesky(cov + jitter * identity)  # [..., L, B, B]

        variance_contribution = cov_cholesky @ tf.cast(self._eps, cov.dtype)  # [..., L, B, S]

        leading_indices = tf.range(tf.rank(variance_contribution) - 3)
        absolute_trailing_indices = [-1, -2, -3] + tf.rank(variance_contribution)
        new_order = tf.concat([leading_indices, absolute_trailing_indices], axis=0)

        return mean[..., None, :, :] + tf.transpose(variance_contribution, new_order)