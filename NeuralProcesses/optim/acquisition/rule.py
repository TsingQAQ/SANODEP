"""
This module contains acquisition rules, which choose the optimal point(s) to query on each step of
the Bayesian optimization process.
"""

from trieste.acquisition.rule import (
    EfficientGlobalOptimization,
    Optional,
    AcquisitionFunction,
    Dataset,
    GreedyAcquisitionFunctionBuilder,
    SingleModelAcquisitionBuilder,
    SingleModelGreedyAcquisitionBuilder,
    SingleModelVectorizedAcquisitionBuilder,
    OBJECTIVE,
)
from .optimizer import (
    init_cond_and_time_batchify_joint,
    automatic_optimizer_selector,
    time_batchify_joint,
    init_cond_and_time_batchify_joint_jax_ver,
    time_batchify_joint_jax_ver,
)
from .function.function import FlaxModelBasedAcqusitionFunction
from trieste.space import Box, LinearConstraint
from trieste.types import TensorType
from functools import partial
from typing import List
import tensorflow as tf
from jax import numpy as np


# this is to ensure that the optimization does not go near the search space boarder, which is likely to cause
# some issue on the ODE solve within the model as it can misunderstood that this point is outside the ODE solver time duration
_search_space_stability_term = 0.0


class MinimumDelayConstrainedBatchEfficientGlobalOptimization(
    EfficientGlobalOptimization
):
    """Implements the Efficient Global Optimization, or EGO, algorithm."""

    def __init__(
        self,
        builder,
        minimum_delta,
        optimizer: Optional[automatic_optimizer_selector] = None,
        num_query_points: List[int] = 1,
        initial_acquisition_function=None,
        acq_optimizer_initial_smp_num: Optional[TensorType] = None,
        acq_optimizer_parallel_num: Optional[int] = None,
        acq_optimizer_max_iter: Optional[int] = None,
    ):
        """
        :param builder: The acquisition function builder to use.
        :param minimum_delta: The minimum time delay between two consecutive query points
        :param optimizer: The optimizer with which to optimize the acquisition function built by
            ``builder``. This should *maximize* the acquisition function, and must be compatible
            with the global search space.
        :param num_query_points: List of int, The number of points to acquire, this is needed to
            initialize optimizers for different batch sizes
        :param initial_acquisition_function: The initial acquisition function to use. Defaults
            to using the builder to construct one, but passing in a previously constructed
            function can occasionally be useful (e.g. to preserve random seeds).
        """

        if np.any(num_query_points <= 0):
            raise ValueError(
                f"Number of query points must be greater than 0, got {num_query_points}"
            )

        if optimizer is None:
            if isinstance(builder, FlaxModelBasedAcqusitionFunction):
                optimizer = partial(automatic_optimizer_selector, is_jax=True, 
                                    initial_samples_num = acq_optimizer_initial_smp_num, 
                                    num_runs = acq_optimizer_parallel_num,
                                    max_iter= acq_optimizer_max_iter)
                self._is_jax = True
            else:
                optimizer = partial(automatic_optimizer_selector, 
                                    initial_samples_num = acq_optimizer_initial_smp_num, 
                                    num_runs = acq_optimizer_parallel_num,
                                    max_iter= acq_optimizer_max_iter)
                self._is_jax = False

        if isinstance(
            builder,
            (
                SingleModelAcquisitionBuilder,
                SingleModelGreedyAcquisitionBuilder,
                SingleModelVectorizedAcquisitionBuilder,
            ),
        ):
            builder = builder.using(OBJECTIVE)

        self._builder = builder
        self._optimizer = optimizer
        self._num_query_points = num_query_points
        self._acquisition_function: Optional[AcquisitionFunction] = (
            initial_acquisition_function
        )
        self._minimum_delta = minimum_delta

    def update_num_query_points(self, num_query_points: List[int]):
        self._num_query_points = num_query_points

    def acquire(
        self,
        t_bounds,
        model,
        initial_loc_bounds: Optional[TensorType] = None,
        initial_loc: Optional[TensorType] = None,
        last_obs_time: Optional[TensorType] = None,
        datasets: Optional[Dataset] = None,
    ):
        """
        Return the query point(s) that optimizes the acquisition function produced by ``builder``
        (see :meth:`__init__`).

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model for each tag.
        :param datasets: The known observer query points and observations. Whether this is required
            depends on the acquisition function used.
        :return: The single (or batch of) points to query.
        """
        # print(f't_bounds: {t_bounds}')
        # print(f'last_obs_time: {last_obs_time}')
        # print(f'num_query_points: {self._num_query_points}')
        updated_optimizer =  lambda _space, _target_func: \
            self._optimizer(_space, 
                            initial_condition_search_space = Box(*initial_loc_bounds) if initial_loc_bounds is not None else None, 
                            time_search_space = Box(np.atleast_1d(t_bounds[0]), np.atleast_1d(t_bounds[1])) if not last_obs_time else Box(np.atleast_1d(t_bounds[0]) + self._minimum_delta, np.atleast_1d(t_bounds[1])), 
                            target_func = _target_func, 
                            is_search_initial_cond = True if initial_loc_bounds is not None else False,
                            )
        # self._optimizer = updated_optimizer
        assert initial_loc_bounds is not None or initial_loc is not None and not (initial_loc_bounds is not None and initial_loc is not None), \
            ValueError("Either initial location is specified or the initial location bounds are specified, not both.")
        if initial_loc_bounds is not None:
            state_dim = len(initial_loc_bounds[0])
        else:
            state_dim = initial_loc.shape[-1]

        different_batch_points = (
            []
        )  # stor optmized inputs of different batch sizes
        different_batch_vals = (
            []
        )  # store optimized acquisition function values of different batch sizes 
        # construct the optimizer
        if initial_loc_bounds is not None: # ooptimize the initial condition, here we need to make use of the multi-start optimizer
            if self._is_jax:
                optimizer = [
                    init_cond_and_time_batchify_joint_jax_ver(
                        updated_optimizer, int(_num_query_points), state_dim
                    )  # note this num_query_points only correspond to the time step number excluding t0
                    for _num_query_points in self._num_query_points
                ]
            else:
                optimizer = [
                    init_cond_and_time_batchify_joint(
                        updated_optimizer, int(_num_query_points), state_dim
                    )
                    for _num_query_points in self._num_query_points
                ]
        else:  # ooptimize the time scheduling
            if self._is_jax:
                optimizer = [
                    time_batchify_joint_jax_ver(updated_optimizer, int(_num_query_points))
                    for _num_query_points in self._num_query_points
                ]
            else:
                optimizer = [
                    time_batchify_joint(updated_optimizer, int(_num_query_points))
                    for _num_query_points in self._num_query_points
                ]
        # loop over query points number
        for _idx, _num_query_points in enumerate(self._num_query_points):
            # rebuild graph each time
            # if self._acquisition_function is None:
            _num_query_points = int(_num_query_points)

            # if this batch size is only used for time, maybe we do not need to add this state dim
            if initial_loc is None:
                # batch_size = _num_query_points + state_dim # if initial location is not provided, we need to optimize the initial location together with the times
                time_batch_size = _num_query_points + 1 # the 1st corresponds to t0
            else:
                # batch_size = _num_query_points
                time_batch_size = _num_query_points	

            self._acquisition_function = self._builder.prepare_acquisition_function(
                model,
                datasets=datasets,
                batch_size = time_batch_size,
            )
            # construct the search space
            # refer https://secondmind-labs.github.io/trieste/3.0.0/notebooks/explicit_constraints.html#Explicit-constraints
            # Create a matrix A with -1 on the diagonal and 1 on the diagonal above it
            if ( # optimize initial location together with times
                initial_loc_bounds is not None
            ):  # optimize initial location together with times, note that, t0 is not explicitly included in constraints because it is not treated as a desgn variblae

                # construct the minimum delay constraint
                A = tf.cast(
                    tf.linalg.diag([-1.0] * (_num_query_points + state_dim))
                    + tf.linalg.diag([1.0] * (_num_query_points + state_dim - 1), k=1), # k represent diagonal shift to the right by initial_state_dim
                    dtype=tf.float64,
                ) # the A's last axis dim shall correspond to the optimization input dim

                # Remove the first initial_state_dim row of A and last row of A 
                A = A[state_dim:-1]

                g_lb = tf.constant([self._minimum_delta] * (_num_query_points - 1), dtype=tf.float64)
                g_ub = tf.constant([1e10] * (_num_query_points - 1), dtype=tf.float64)
                aug_lb = tf.cast(
                    tf.concat(
                        [
                            initial_loc_bounds[0], # initial location lower bounds
                            tf.repeat( # repeat the lower bound of the time
                                tf.expand_dims(t_bounds[0], axis=-1)
                                + self._minimum_delta,
                                _num_query_points,
                                axis=-1,
                            ),
                        ],
                        axis=-1,
                    ),
                    dtype=tf.float64,
                )
                aug_ub = tf.cast(
                    tf.concat(
                        [
                            initial_loc_bounds[1],
                            tf.repeat(
                                tf.expand_dims(t_bounds[1], axis=-1),
                                _num_query_points,
                                axis=-1,
                            )
                            - _search_space_stability_term,
                        ],
                        axis=-1,
                    ),
                    dtype=tf.float64,
                )

                aug_search_space = Box(
                    aug_lb, aug_ub, constraints=[LinearConstraint(A, g_lb, g_ub)]
                )

                greedy = isinstance(self._builder, GreedyAcquisitionFunctionBuilder)
                with tf.name_scope("EGO.optimizer" + "[0]" * greedy):
                    points, val = optimizer[_idx](aug_search_space, t_bounds[0], self._acquisition_function)

            elif initial_loc is not None: # optimize only the time scheduling
                assert last_obs_time is not None

                aug_lb = tf.cast(
                    tf.repeat(
                        tf.expand_dims(last_obs_time, axis=-1) + self._minimum_delta,
                        _num_query_points,
                        axis=-1,
                    ),
                    dtype=tf.float64,
                )
                aug_ub = tf.cast(
                    tf.repeat(
                        tf.expand_dims(t_bounds[1], axis=-1), _num_query_points, axis=-1
                    )
                    - _search_space_stability_term,
                    dtype=tf.float64,
                )

                # print(f'aug_lb: {aug_lb}')
                # print(f'aug_ub: {aug_ub}')
                # print(f'num_query_points: {_num_query_points}')

                if _num_query_points != 1:
                    # print('Work on more than 1 query')
                    A = tf.cast(
                        tf.linalg.diag([-1.0] * (_num_query_points))
                        + tf.linalg.diag([1.0] * (_num_query_points - 1), k=1),
                        dtype=tf.float64,
                    )

                    # Remove the last row of A and the first row of A
                    A = A[:-1, :]
                    g_lb = tf.constant(
                        [self._minimum_delta] * (_num_query_points - 1),
                        dtype=tf.float64,
                    )
                    g_ub = tf.constant(
                        [1e10] * (_num_query_points - 1), dtype=tf.float64
                    )

                    aug_search_space = Box(
                        aug_lb, aug_ub, constraints=[LinearConstraint(A, g_lb, g_ub)]
                    )
                else:  # only one point left to search, not use linear constraint but simply use aug_lb to incoperate this boundary constraint
                    # print('Work on only 1 query')
                    aug_search_space = Box(aug_lb, aug_ub)

                greedy = isinstance(self._builder, GreedyAcquisitionFunctionBuilder)
                with tf.name_scope("EGO.optimizer" + "[0]" * greedy):
                    points, val = optimizer[_idx](np.asarray(initial_loc.numpy()),expanded_search_space=aug_search_space,f=self._acquisition_function,)
            else:
                raise NotImplementedError
            # different_batch_points.append(points)
            different_batch_points.append(tf.cast(points, dtype=tf.float64))
            # different_batch_vals.append(val)
            different_batch_vals.append(tf.cast(val, dtype=tf.float64))
            
        idx = tf.argmax(different_batch_vals)
        points = different_batch_points[idx]

        return points
