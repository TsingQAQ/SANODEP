from trieste.acquisition.function.multi_objective import (
    HasReparamSampler,
    Sequence,
    TensorType,
    Callable,
    get_reference_point,
    DEFAULTS,
    tf,
    Optional,
    Dataset,
    AcquisitionFunction,
    cast,
    Pareto,
    prepare_default_non_dominated_partition_bounds,
    ReparametrizationSampler,
    combinations,
)
from ..interface import CustomizedSingleModelAcquisitionBuilder

import jax
from jax import random
import tensorflow as tf
from jax import numpy as np
from jax.typing import ArrayLike
from NeuralProcesses.data.datasets import NP_Dataset
from .function import FlaxModelBasedAcqusitionFunction
from trieste.acquisition.function.function import (
    AcquisitionFunction,
    AcquisitionFunctionClass,
)
from NeuralProcesses.models.utils.transformation import IdentityTransform


class GreyBoxBatchMonteCarloExpectedHypervolumeImprovement(
    CustomizedSingleModelAcquisitionBuilder
):
    """
    Grey box batch monte carlo expected hypervolume improvement acquisition function.

    NOTE:
    1. since the multi-objective acq in trieste aims for mimimize, we hence tailor
    this acquisition to maximize the first objective function and minimize the second one.
    2. due to the specific design of optimization scheme, because different batch sizes
    are actually informed in the prepare_acquisition_function, we bring gen_q_subset_indices
    into the acquisition function to avoid the need redundant calculation for speed consideration
    """

    def __init__(
        self,
        sample_size: int,
        reference_point_spec: (
            Sequence[float] | TensorType | Callable[..., TensorType]
        ) = get_reference_point,
        *,
        obj_func_form: Callable,
        jitter: float = DEFAULTS.JITTER,
        initial_condition_mapping: Optional[Callable] = IdentityTransform(),
        time_scaling: float = 1.0,
        
    ):
        """
        :param sample_size: The number of samples from model predicted distribution for
            each batch of points.
        :param reference_point_spec: this method is used to determine how the reference point is
            calculated. If a Callable function specified, it is expected to take existing
            posterior mean-based observations (to screen out the observation noise) and return
            a reference point with shape [D] (D represents number of objectives). If the Pareto
            front location is known, this arg can be used to specify a fixed reference point
            in each bo iteration. A dynamic reference point updating strategy is used by
            default to set a reference point according to the datasets.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive, or
            ``jitter`` is negative.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_greater_equal(jitter, 0.0)
        self._sample_size = sample_size
        self._jitter = jitter
        if callable(reference_point_spec):
            self._ref_point_spec: tf.Tensor | Callable[..., TensorType] = (
                reference_point_spec
            )
        else:
            self._ref_point_spec = tf.convert_to_tensor(reference_point_spec)

        self._ref_point = None
        self._obj_func_form = obj_func_form
        self._init_cond_map = initial_condition_mapping
        self._time_scaling = time_scaling

    def prepare_acquisition_function(
        self,
        model: HasReparamSampler,
        dataset: Optional[Dataset] = None,
        batch_size: int = 1,
    ) -> AcquisitionFunction:
        """
        :param model: The model. Must have event shape [1].
        :param dataset: The data from the observer. Must be populated.
        :return: The batch expected hypervolume improvement acquisition function.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        mean, _ = model.predict(dataset.query_points)
        aggregate_obj_mean = self._obj_func_form(mean)
        # the minus is for maximization of the first objective
        mean = tf.concat([-aggregate_obj_mean, dataset.query_points[..., -1:] * self._time_scaling], axis=-1)

        if callable(self._ref_point_spec):
            self._ref_point = tf.cast(self._ref_point_spec(mean), dtype=mean.dtype)
        else:
            self._ref_point = tf.cast(self._ref_point_spec, dtype=mean.dtype)

        _pf = Pareto(mean)
        screened_front = _pf.front[tf.reduce_all(_pf.front <= self._ref_point, -1)]
        # prepare the partitioned bounds of non-dominated region for calculating of the
        # hypervolume improvement in this area
        _partition_bounds = prepare_default_non_dominated_partition_bounds(
            self._ref_point, screened_front
        )

        if not isinstance(model, HasReparamSampler):
            raise ValueError(
                f"The batch Monte-Carlo expected hyper-volume improvement function only supports "
                f"models that implement a reparam_sampler method; received {model!r}"
            )

        sampler = model.reparam_sampler(self._sample_size)

        def gen_q_subset_indices(q: int) -> tf.RaggedTensor:
            # generate all subsets of [1, ..., q] as indices
            indices = list(range(q))
            return tf.ragged.constant(
                [list(combinations(indices, i)) for i in range(1, q + 1)]
            )

        q_subset_indices = gen_q_subset_indices(batch_size)
        return batch_ehvi(
            sampler,
            self._jitter,
            _partition_bounds,
            q_subset_indices,
            obj_func_form=self._obj_func_form,
            initial_cond_mapper=self._init_cond_map,
            time_scaling = self._time_scaling
        )


# will this function run faster if we jit it somehow?
def batch_ehvi(
    sampler: ReparametrizationSampler[HasReparamSampler],
    sampler_jitter: float,
    partition_bounds: tuple[TensorType, TensorType],
    q_subset_indices: TensorType,
    obj_func_form: Optional[None] = None,
    initial_cond_mapper: Optional[Callable] = IdentityTransform(),
    time_scaling: float = 1.0,
) -> AcquisitionFunction:
    """
    Note that this may have OOM issue because of the initial random sample in multi-start-bfgs

    :param sampler: The posterior sampler, which given query points `at`, is able to sample
        the possible observations at 'at'.
    :param sampler_jitter: The size of the jitter to use in sampler when stabilising the Cholesky
        decomposition of the covariance matrix.
    :param partition_bounds: with shape ([N, D], [N, D]), partitioned non-dominated hypercell
        bounds for hypervolume improvement calculation
    :return: The batch expected hypervolume improvement acquisition
        function for objective minimisation.
    """

    def acquisition(at: TensorType) -> TensorType:
        _batch_size = at.shape[-2]  # B
        raw_query_points = tf.concat([tf.convert_to_tensor(initial_cond_mapper(at[..., :-1]), dtype=at.dtype), at[..., -1:]], axis=-1)
        samples = sampler.sample(raw_query_points, jitter=sampler_jitter)  # [..., S, B, num_obj]
        aggregated_samples = obj_func_form(samples)  # [..., S, B, 1]
        dummy_obj = tf.repeat(
            tf.expand_dims(at[..., -1:], axis=-3),
            axis=-3,
            repeats=aggregated_samples.shape[-3],
        )  # [..., S, B]
        samples = tf.concat([-aggregated_samples, dummy_obj * time_scaling], axis=-1)  # [..., S, B, 2]

        # q_subset_indices = gen_q_subset_indices(_batch_size)

        hv_contrib = tf.zeros(tf.shape(samples)[:-2], dtype=samples.dtype)
        lb_points, ub_points = partition_bounds

        def hv_contrib_on_samples(
            obj_samples: TensorType,
        ) -> TensorType:  # calculate samples overlapped area's hvi for obj_samples
            # [..., S, Cq_j, j, num_obj] -> [..., S, Cq_j, num_obj]
            overlap_vertices = tf.reduce_max(obj_samples, axis=-2)

            overlap_vertices = (
                tf.maximum(  # compare overlap vertices and lower bound of each cell:
                    tf.expand_dims(overlap_vertices, -3),  # expand a cell dimension
                    lb_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :],
                )
            )  # [..., S, K, Cq_j, num_obj]

            lengths_j = tf.maximum(  # get hvi length per obj within each cell
                (
                    ub_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :]
                    - overlap_vertices
                ),
                0.0,
            )  # [..., S, K, Cq_j, num_obj]

            areas_j = tf.reduce_sum(  # sum over all subsets Cq_j -> [..., S, K]
                tf.reduce_prod(lengths_j, axis=-1), axis=-1  # calc hvi within each K
            )

            return tf.reduce_sum(areas_j, axis=-1)  # sum over cells -> [..., S]

        for j in tf.range(1, _batch_size + 1):  # Inclusion-Exclusion loop
            q_choose_j = tf.gather(q_subset_indices, j - 1).to_tensor()
            # gather all combinations having j points from q batch points (Cq_j)
            j_sub_samples = tf.gather(
                samples, q_choose_j, axis=-2
            )  # [..., S, Cq_j, j, num_obj]
            hv_contrib += tf.cast(
                (-1) ** (j + 1), dtype=samples.dtype
            ) * hv_contrib_on_samples(j_sub_samples)

        return tf.reduce_mean(hv_contrib, axis=-1, keepdims=True)  # average through MC

    return acquisition


class GreyBoxBatchMonteCarloExpectedHypervolumeImprovementCompatibleWithFlaxModels(
    GreyBoxBatchMonteCarloExpectedHypervolumeImprovement,
    FlaxModelBasedAcqusitionFunction,
):
    """
    Jax version of the Grey box batch monte carlo expected hypervolume improvement acquisition function.
    """

    def __init__(
        self,
        sample_size: int,
        sample_rng: random.PRNGKey,
        reference_point_spec: (
            Sequence[float] | TensorType | Callable[..., TensorType]
        ) = get_reference_point,
        *,
        obj_func_form: Optional[None] = None,
        jitter: float = DEFAULTS.JITTER,
        trajectory_aware: bool = False,
        initial_condition_mapping: Optional[Callable] = IdentityTransform(),
        time_scaling: float = 1.0,

    ):
        super().__init__(sample_size, reference_point_spec = reference_point_spec, obj_func_form=obj_func_form, jitter=jitter, 
                         initial_condition_mapping=initial_condition_mapping)
        self.trajectory_aware = trajectory_aware
        self._sample_rng = sample_rng
        self._time_scaling = time_scaling

    def prepare_acquisition_function(
        self,
        model: HasReparamSampler,
        dataset: Optional[NP_Dataset] = None,
        batch_size: int = 1,
    ) -> AcquisitionFunction:
        """
        :param model: The model. Must have event shape [1].
        :param dataset: The data from the observer. Must be populated.
        :return: The batch *expected improvement* acquisition function.
        :raise ValueError (or InvalidArgumentError): If ``dataset`` is not populated, or ``model``
            does not have an event shape of [1].
        """
        dataset = dataset.formalize_training_data_for_trieste(tf.float64)
        # mean, _ = model.predict(np.asarray(dataset.query_points.numpy()))

        # note that because Jax model has its internal initial cond mapper, so here we will have to map it back to the zipped case
        query_points = np.asarray(dataset.query_points.numpy())
        mean, _ = model.predict(np.concatenate([self._init_cond_map.backward(query_points[..., :-1]), query_points[..., -1:]], axis=-1))
        
        aggregate_obj_mean = self._obj_func_form(mean)
        if self.trajectory_aware:
            aggregate_obj_mean = np.squeeze(aggregate_obj_mean, axis=-2)
        # aggregate_obj_mean = np.squeeze(
        #     self._obj_func_form(mean), axis=-2
        # )  # [B, S, D] -> [B, S]
        # the minus is for maximization of the first objective
        mean = np.concatenate(
            [-aggregate_obj_mean, dataset.query_points[..., -1:].numpy() * self._time_scaling], axis=-1
        )

        if callable(self._ref_point_spec):
            self._ref_point = tf.cast(
                self._ref_point_spec(mean), dtype=mean.dtype
            ).numpy()
        else:
            self._ref_point = tf.cast(self._ref_point_spec, dtype=mean.dtype)

        _pf = Pareto(mean)
        screened_front = _pf.front[tf.reduce_all(_pf.front <= self._ref_point, -1)]
        # prepare the partitioned bounds of non-dominated region for calculating of the
        # hypervolume improvement in this area
        _partition_bounds = prepare_default_non_dominated_partition_bounds(
            self._ref_point, screened_front
        )
        _partition_bounds = (_partition_bounds[0].numpy(), _partition_bounds[1].numpy())

        def gen_q_subset_indices(q: int) -> tf.RaggedTensor:
            # generate all subsets of [1, ..., q] as indices
            indices = list(range(q))
            return [np.asarray(list(combinations(indices, i))) for i in range(1, q + 1)]

        q_subset_indices = gen_q_subset_indices(batch_size)
        # q_subset_indices_TF = gen_q_subset_indices_tf(batch_size)
        return batch_monte_carlo_expected_improvement_compatible_with_flax_models(
            model,
            _partition_bounds,
            q_subset_indices,
            self._sample_size, 
            self._sample_rng,
            batch_size=batch_size,
            obj_func_form=self._obj_func_form,
            time_scaling = self._time_scaling
        ).get_jittable_acq_func()


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
        partitioned_bounds: list,
        q_subset_indices: TensorType,
        sample_size: int, 
        sample_rng: random.PRNGKey,
        batch_size,
        obj_func_form: Optional[None] = None,
        time_scaling: float = 1.0,
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
        self.obj_func_form = obj_func_form
        self.partition_bounds = partitioned_bounds
        self.q_subset_indices = q_subset_indices
        # self.q_subset_indices_TF = q_subset_indices_TF
        self._batch_size = batch_size
        self._iep_loop_array = np.arange(1, self._batch_size + 1)
        self._sample_size = sample_size
        self._sample_rng = sample_rng
        self._time_scaling = time_scaling

    def get_jittable_acq_func(self):
        @jax.jit
        def single_eval(_x):
            samples = self._sampler(_x, self._sample_size, self._sample_rng)  # [timesteps, state]
            aggregated_samples = self.obj_func_form(samples)  # [..., S, B, 1]

            dummy_obj = np.repeat(
                np.expand_dims(_x[..., -aggregated_samples.shape[-2] :], axis=-2),
                axis=-2,
                repeats=aggregated_samples.shape[-3],
            )  # [..., S, 1, 1]
            samples = np.concatenate(
                [-aggregated_samples, dummy_obj[..., None] * self._time_scaling], axis=-1
            )  # [..., S, B, 2]

            hv_contrib = np.zeros(np.shape(samples)[:-2])
            lb_points, ub_points = self.partition_bounds

            def hv_contrib_on_samples_jaxjit(
                obj_samples: ArrayLike,
            ) -> ArrayLike:
                # [..., S, Cq_j, j, num_obj] -> [..., S, Cq_j, num_obj]
                overlap_vertices = np.max(obj_samples, axis=-2)

                overlap_vertices = np.maximum(  # compare overlap vertices and lower bound of each cell:
                    np.expand_dims(overlap_vertices, -3),  # expand a cell dimension
                    lb_points[None, None, :, None, :],
                )  # [..., S, K, Cq_j, num_obj]

                lengths_j = np.maximum(  # get hvi length per obj within each cell
                    (ub_points[None, None, :, None, :] - overlap_vertices), 0.0
                )  # [..., S, K, Cq_j, num_obj]

                areas_j = np.sum(  # sum over all subsets Cq_j -> [..., S, K]
                    np.prod(lengths_j, axis=-1), axis=-1  # calc hvi within each K
                )

                return np.sum(areas_j, axis=-1)  # sum over cells -> [..., S]

            def hv_contrib_on_samples(
                obj_samples: TensorType,
            ) -> TensorType:  # calculate samples overlapped area's hvi for obj_samples
                # [..., S, Cq_j, j, num_obj] -> [..., S, Cq_j, num_obj]
                overlap_vertices = tf.reduce_max(obj_samples, axis=-2)

                overlap_vertices = tf.maximum(  # compare overlap vertices and lower bound of each cell:
                    tf.expand_dims(overlap_vertices, -3),  # expand a cell dimension
                    lb_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :],
                )  # [..., S, K, Cq_j, num_obj]

                lengths_j = tf.maximum(  # get hvi length per obj within each cell
                    (
                        ub_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :]
                        - overlap_vertices
                    ),
                    0.0,
                )  # [..., S, K, Cq_j, num_obj]

                areas_j = tf.reduce_sum(  # sum over all subsets Cq_j -> [..., S, K]
                    tf.reduce_prod(lengths_j, axis=-1),
                    axis=-1,  # calc hvi within each K
                )

                return tf.reduce_sum(areas_j, axis=-1)  # sum over cells -> [..., S]

            # for j in np.arange(1, self._batch_size + 1):  # Inclusion-Exclusion loop
            for j, q_choose_j in enumerate(self.q_subset_indices):
                j_sub_samples = np.take(
                    samples, q_choose_j, axis=-2
                )  # [..., S, Cq_j, j, num_obj]
                hv_contrib += (-1) ** j * hv_contrib_on_samples_jaxjit(j_sub_samples)

            return np.mean(hv_contrib, axis=-1, keepdims=True)  # average through MC

        return single_eval

    def __call__(self, x: ArrayLike) -> ArrayLike:
        pass
