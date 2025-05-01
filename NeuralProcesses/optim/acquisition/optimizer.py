"""
Acquisition Function Optimizer
==============================
There is a known bug in Scipy Optimize that the trust-constr method (used in time scheduling optimization) can 
have evaluations outside the bounds. As a result, when optimization has failed, it can have result that is outside 
the specified design space that may subsueqently cause problems (e.g., a negative time schedule that can cause failure 
of diffeqsolve). This is a issue from Scipy itself. It is believed to be fixed in  https://github.com/scipy/scipy/pull/21257 
in a not yet released version (1.15).
"""
import numpy as np
from jax import jit
from jax import vmap
from jax import numpy as jnp
from trieste.space import Box
from jax import value_and_grad
from typing import Optional
from jax.typing import ArrayLike


from typing import List, Iterable
import tensorflow as tf
from trieste.acquisition.optimizer import (
    SearchSpace,
    Union,
    AcquisitionFunction,
    Tuple,
    TensorType,
    DiscreteSearchSpace,
    CollectionSearchSpace,
    optimize_discrete,
    NUM_SAMPLES_MIN,
    NUM_SAMPLES_DIM,
    NUM_RUNS_DIM,
    AcquisitionOptimizer,
    SearchSpaceType,
    InitialPointSampler,
    Any,
    generate_initial_points,
    logging,
    get_bounds_of_box_relaxation_around_point,
    TaggedProductSearchSpace,
    TaggedMultiSearchSpace,
    spo,
    ScipyOptimizerGreenlet,
    _perform_parallel_continuous_optimization,
)


def ode_aware_sample_from_space(initial_cond_search_space, time_search_space, num_samples: int, batch_size: Optional[int] = None) -> InitialPointSampler:
    """
    An initial point sampler that returns `num_samples` points. If `batch_size` is specified,
    then these are returned in batches of that size, to preserve memory usage.

    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    if isinstance(batch_size, int) and batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    batch_size_int = batch_size or num_samples
    initial_cond_search_dim = initial_cond_search_space.lower.shape[0]

    def sampler(space: Box) -> Iterable[TensorType]:
        for offset in range(0, num_samples, batch_size_int):
            x0_smps = initial_cond_search_space.sample(min(num_samples - offset, batch_size_int))
            _obs_times = space.lower.shape[0] - initial_cond_search_dim # the number of observation times
            # modification notice: because time_search_space does not consider the initial condition is given internally, 
            # use the original linspace will generate a duplicate of the initial condition, so we need to skip the first one
            # so now we go for a bit hacky way as staring from initial_cond_search_dim the space is the time space, we use 
            # the first one as the initial condition, and the last one as termination time
            t_schedule = tf.repeat(tf.linspace(tf.squeeze(space.lower[initial_cond_search_dim]), tf.squeeze(space.upper[-1]), _obs_times)[None, ...], x0_smps.shape[0], axis=0)
            yield np.concatenate([tf.cast(x0_smps, dtype=tf.float64), tf.cast(t_schedule, dtype=tf.float64)], axis=-1)
            # t_shedule = tf.repeat(tf.linspace(tf.squeeze(time_search_space.lower), tf.squeeze(time_search_space.upper), _obs_times)[None, ...], x0_smps.shape[0], axis=0)
            # yield np.concatenate([tf.cast(x0_smps, dtype=tf.float64), tf.cast(t_shedule, dtype=tf.float64)], axis=-1)
            # t_smps = time_search_space.sample(min(num_samples - offset, batch_size_int))
            # yield space.sample(min(num_samples - offset, batch_size_int))

    return sampler


def automatic_optimizer_selector(
    space: SearchSpace,
    time_search_space: Box,
    target_func: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
    initial_condition_search_space: Optional[Box] = None, 
    initial_samples_num: Optional[int] = None, 
    num_runs: Optional[int] = None,
    max_iter: Optional[int] = 100, 
    is_jax: bool = False,
    is_search_initial_cond: bool = False
) -> TensorType:
    """
    A wrapper around our :const:`AcquisitionOptimizer`s. This class performs
    an :const:`AcquisitionOptimizer` appropriate for the
    problem's :class:`~trieste.space.SearchSpace`.

    :param space: The space of points over which to search, for points with shape [D].
    :param target_func: The function to maximise, with input shape [..., 1, D] and output shape
            [..., 1].
    :param num_runs number of optimization runs
    :param is_search_initial_cond: whether to search for the initial condition, if set to True, 
            the initial condition search space will be used as the search space
    :return: The batch of points in ``space`` that maximises ``target_func``, with shape [1, D].
    """

    if isinstance(space, DiscreteSearchSpace):
        return optimize_discrete(space, target_func)

    elif isinstance(space, (Box, CollectionSearchSpace)):
        space_dim = space.dimension
        if initial_samples_num is None:
            num_samples = tf.maximum(NUM_SAMPLES_MIN, NUM_SAMPLES_DIM * space_dim)
            num_samples = tf.minimum(num_samples, 500)
        else:
            num_samples = initial_samples_num
        if num_runs is None:
            num_runs = tf.minimum(NUM_RUNS_DIM * space_dim, 10)
        points, val = generate_continuous_optimizer(
            initial_condition_search_space = initial_condition_search_space, 
            time_search_space = time_search_space,
            num_initial_samples=num_samples,
            num_optimization_runs=num_runs,
            is_jax=is_jax,
            is_search_initial_cond = is_search_initial_cond,
            optimizer_args={"options": {"maxiter": max_iter}},
        )(space, target_func)
        return points, val

    else:
        raise NotImplementedError(
            f""" No optimizer currently supports acquisition function
                    maximisation over search spaces of type {space}.
                    Try specifying the optimize_random optimizer"""
        )


def generate_continuous_optimizer(
    initial_condition_search_space: Box, 
    time_search_space: Box,
    num_initial_samples: int | InitialPointSampler = NUM_SAMPLES_MIN,
    num_optimization_runs: int = 10,
    num_recovery_runs: int = 10,
    optimizer_args: Optional[dict[str, Any]] = None,
    is_jax: bool = False,
    is_search_initial_cond: bool = False
) -> AcquisitionOptimizer[Box | CollectionSearchSpace]:
    """
    Generate a gradient-based optimizer for :class:'Box' and :class:'CollectionSearchSpace'
    spaces and batches of size 1. In the case of a :class:'CollectionSearchSpace', We perform
    gradient-based optimization across all :class:'Box' subspaces, starting from the best location
    found across a sample of `num_initial_samples` random points.

    We advise the user to either use the default `NUM_SAMPLES_MIN` for `num_initial_samples`, or
    `NUM_SAMPLES_DIM` times the dimensionality of the search space, whichever is greater.
    Similarly, for `num_optimization_runs`, we recommend using `NUM_RUNS_DIM` times the
    dimensionality of the search space.

    This optimizer uses Scipy's L-BFGS-B optimizer. We run `num_optimization_runs` separate
    optimizations in parallel, each starting from one of the best `num_optimization_runs` initial
    query points.

    If all `num_optimization_runs` optimizations fail to converge then we run
    `num_recovery_runs` additional runs starting from random locations (also ran in parallel).

    **Note:** using a large number of `num_initial_samples` and `num_optimization_runs` with a
    high-dimensional search space can consume a large amount of CPU memory (RAM).

    :param num_initial_samples: The starting point(s) of the optimization. This can be either
        the number of random samples to use, or a function that given the search space returns
        the points to use. The latter can be used for example to add pre-optimized starting points
        to the random points, as well as to batch point generation to reduce memory usage for
        high-dimensional problems.
    :param num_optimization_runs: The number of separate optimizations to run.
    :param num_recovery_runs: The maximum number of recovery optimization runs in case of failure.
    :param optimizer_args: The keyword arguments to pass to the Scipy L-BFGS-B optimizer.
        Check `minimize` method  of :class:`~scipy.optimize` for details of which arguments
        can be passed. Note that method, jac and bounds cannot/should not be changed.
    :return: The acquisition optimizer.
    """
    if num_optimization_runs <= 0:
        raise ValueError(
            f"num_optimization_runs must be positive, got {num_optimization_runs}"
        )

    if (
        not callable(num_initial_samples)
        and num_initial_samples < num_optimization_runs
    ):
        raise ValueError(
            f"""
            num_initial_samples {num_initial_samples} must be at
            least num_optimization_runs {num_optimization_runs}
            """
        )

    if num_recovery_runs < 0:
        raise ValueError(
            f"num_recovery_runs must be zero or greater, got {num_recovery_runs}"
        )
    
    if not is_search_initial_cond:
        num_optimization_runs = 1 # I am not sure if this will be really used
    else:
        initial_sampler = (
            ode_aware_sample_from_space(initial_condition_search_space, time_search_space, num_initial_samples)
            if not callable(num_initial_samples)
            else num_initial_samples
        )

    def optimize_continuous(
        space: Box | CollectionSearchSpace,
        target_func: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
    ) -> TensorType:
        """
        A gradient-based :const:`AcquisitionOptimizer` for :class:'Box'
        and :class:`CollectionSearchSpace' spaces.

        For :class:'CollectionSearchSpace' we only apply gradient updates to
        its class:'Box' subspaces.

        When this function receives an acquisition-integer tuple as its `target_func`,it
        optimizes each of the individual V functions making up `target_func`, i.e.
        evaluating `num_initial_samples` samples, running `num_optimization_runs` runs, and
        (if necessary) running `num_recovery_runs` recovery run for each of the individual
        V functions.

        :param space: The space over which to search.
        :param target_func: The function to maximise, with input shape [..., V, D] and output shape
                [..., V].
        :return: The V points in ``space`` that maximises``target_func``, with shape [V, D].
        """

        if isinstance(target_func, tuple):  # check if we need a vectorized optimizer
            target_func, V = target_func
        else:
            V = 1

        if V <= 0:
            raise ValueError(f"vectorization must be positive, got {V}")
        if is_search_initial_cond:
            if is_jax:
                initial_points = generate_initial_points_jax_ver(
                    num_optimization_runs, initial_sampler, space, target_func, V
                )  # [num_optimization_runs,V,D]
            else:
                initial_points = generate_initial_points(
                    num_optimization_runs, initial_sampler, space, target_func, V
                )  # [num_optimization_runs,V,D]
        else:
            # initial_cond_search_dim = initial_condition_search_space.lower.shape[0]
            # _obs_times = space.lower.shape[0] - initial_cond_search_dim
            _obs_times = space.lower.shape[0]
            # below is not true, because time_search_sapce lower is not the one with delta considered
            # t_shedule = tf.linspace(tf.squeeze(time_search_space.lower), tf.squeeze(time_search_space.upper), _obs_times)
            t_shedule = tf.linspace(tf.squeeze(time_search_space.lower), tf.squeeze(time_search_space.upper), _obs_times)
            initial_points = t_shedule[None, None, ...]

        if len(initial_points) < num_optimization_runs:
            raise ValueError(
                f"Not enough initial points generated ({len(initial_points)} "
                f"for {num_optimization_runs} optimization runs)"
            )
        if not is_jax:
            (
                successes,
                fun_values,
                chosen_x,
                nfev,
            ) = _perform_parallel_continuous_optimization(  # [num_optimization_runs, V]
                target_func,
                space,
                initial_points,
                optimizer_args or {},
            )
        else:
            # print(space)
            (
                successes,
                fun_values,
                chosen_x,
                nfev,
            ) = _perform_parallel_continuous_optimization_jax_ver(  # [num_optimization_runs, V]
                target_func,
                space,
                initial_points,
                optimizer_args or {},
            )

        successful_optimization = tf.reduce_all(
            tf.reduce_any(successes, axis=0)
        )  # Check that at least one optimization was successful for each function
        total_nfev = tf.reduce_max(
            nfev
        )  # acquisition function is evaluated in parallel

        recovery_run = False  # Do we need this part
        if num_recovery_runs and not successful_optimization:
            # if all optimizations failed for a function then try again from random starts
            random_points = space.sample(num_recovery_runs)
            if tf.rank(random_points) == 3:
                # If samples is a tensor of rank 3, then it is a batch of samples. In this case
                # the vectorization of the target function must be a multiple of the length of the
                # second (batch) dimension.
                remainder = V % tf.shape(random_points)[1]
                tf.debugging.assert_equal(
                    remainder,
                    tf.cast(0, dtype=remainder.dtype),
                    message=(
                        f"""
                        The vectorization of the target function {V} must be a multiple of the batch
                        shape of random samples {tf.shape(random_points)[1]}.
                        """
                    ),
                )
                multiple = V // tf.shape(random_points)[1]
                tiled_random_points = tf.tile(
                    random_points, [1, multiple, 1]  # [num_recovery_runs, V, D]
                )
            else:
                tf.debugging.assert_rank(
                    random_points,
                    2,
                    message=(
                        f"""
                        The random samples must be a tensor of rank 2, got a tensor of rank
                        {tf.rank(random_points)}.
                        """
                    ),
                )
                tiled_random_points = tf.tile(
                    random_points[:, None, :], [1, V, 1]
                )  # [num_recovery_runs, V, D]

            if not is_jax:
                (
                    recovery_successes,
                    recovery_fun_values,
                    recovery_chosen_x,
                    recovery_nfev,
                ) = _perform_parallel_continuous_optimization(
                    target_func, space, tiled_random_points, optimizer_args or {}
                )
            else:
                (
                    recovery_successes,
                    recovery_fun_values,
                    recovery_chosen_x,
                    recovery_nfev,
                ) = _perform_parallel_continuous_optimization_jax_ver(  # [num_optimization_runs, V]
                    target_func,
                    space,
                    tiled_random_points,
                    optimizer_args or {},
                )

            successes = tf.concat(
                [successes, recovery_successes], axis=0
            )  # [num_optimization_runs + num_recovery_runs, V]
            fun_values = tf.cast(fun_values, tf.float64)
            recovery_fun_values = tf.cast(recovery_fun_values, tf.float64)
            fun_values = tf.concat(
                [fun_values, recovery_fun_values], axis=0
            )  # [num_optimization_runs + num_recovery_runs, V]
            chosen_x = tf.cast(chosen_x, tf.float64)
            recovery_chosen_x = tf.cast(recovery_chosen_x, tf.float64)
            chosen_x = tf.concat(
                [chosen_x, recovery_chosen_x], axis=0
            )  # [num_optimization_runs + num_recovery_runs, V, D]

            successful_optimization = tf.reduce_all(
                tf.reduce_any(successes, axis=0)
            )  # Check that at least one optimization was successful for each function
            total_nfev += tf.reduce_max(tf.cast(recovery_nfev, total_nfev.dtype))
            recovery_run = True

        if not successful_optimization:  # return error if still failed
            print(
                f"""
                  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    Acquisition function optimization failed,
                    even after {num_recovery_runs + num_optimization_runs} restarts.
                  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
                    """
            )

        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                logging.scalar("spo_af_evaluations", total_nfev)
                if recovery_run:
                    logging.text(
                        "spo_recovery_run",
                        f"Acquisition function optimization failed after {num_optimization_runs} "
                        f"optimization runs, requiring recovery runs",
                    )

                _target_func: AcquisitionFunction = target_func  # make mypy happy

                def improvements() -> tf.Tensor:
                    best_initial_values = tf.math.reduce_max(
                        _target_func(initial_points), axis=0
                    )
                    best_values = tf.math.reduce_max(fun_values, axis=0)
                    improve = best_values - tf.cast(
                        best_initial_values, best_values.dtype
                    )
                    return improve[0] if V == 1 else improve

                if V == 1:
                    logging.scalar("spo_improvement_on_initial_samples", improvements)
                else:
                    logging.histogram(
                        "spo_improvements_on_initial_samples", improvements
                    )
        # extract the best points but only based on those satisfying the constraints
        # note that this following line of code is added by myself, this is the way to ensure that the solution satisfying the constraints internally
        if len(space.constraints) != 0:
            feasible_masks = tf.experimental.numpy.atleast_1d(tf.squeeze(tf.logical_and(space.contains(tf.cast(chosen_x, dtype=space.lower.dtype)), space.is_feasible(tf.cast(chosen_x, dtype=space.upper.dtype)))))
        else: # when search the last time point, the constraints are considered in the form of boundary, so we don't need to consider the constraints
            feasible_masks = tf.experimental.numpy.atleast_1d(tf.squeeze(space.contains(tf.cast(chosen_x, dtype=space.lower.dtype))))
        best_run_ids = tf.math.argmax(fun_values[feasible_masks], axis=0)  # [V]
        chosen_points = tf.gather(
            tf.transpose(chosen_x[feasible_masks], [1, 0, 2]), best_run_ids, batch_dims=1
        )  # [V, D]

        return chosen_points, tf.reduce_max(fun_values[feasible_masks])

    return optimize_continuous


def generate_initial_points_jax_ver(
    num_initial_points: int,
    initial_sampler: InitialPointSampler,
    space: SearchSpace,
    target_func: AcquisitionFunction,
    vectorization: int = 1,
) -> TensorType:
    """
    Return the best starting points for an optimization from those generated by a given sampler.

    :param num_initial_points: Number of best starting points to return.
    :param initial_sampler: Initial point sampler.
    :param space: Search space.
    :param target_func: Target function being optimized.
    :param vectorization: Vectorization of the target function.
    """
    top_fun_values: Optional[TensorType] = None  # [V, num_optimization_runs]
    top_candidates: Optional[TensorType] = None  # [V, num_optimization_runs, D]

    for candidates in initial_sampler(space):
        if tf.rank(candidates) == 3:
            # If samples is a tensor of rank 3, then it is a batch of samples. In this case
            # the vectorization of the target function must be a multiple of the length of the
            # second (batch) dimension.
            remainder = vectorization % tf.shape(candidates)[1]
            tf.debugging.assert_equal(
                remainder,
                tf.cast(0, dtype=remainder.dtype),
                message=(
                    f"""
                    The vectorization of the target function {vectorization} must be a multiple of
                    the batch shape of initial samples {tf.shape(candidates)[1]}.
                    """
                ),
            )
            multiple = vectorization // tf.shape(candidates)[1]
            tiled_candidates = tf.tile(candidates, [1, multiple, 1])  # [samples, V, D]
        else:
            tf.debugging.assert_rank(
                candidates,
                2,
                message=(
                    f"""
                    The initial samples must be a tensor of rank 2, got a tensor of rank
                    {tf.rank(candidates)}.
                    """
                ),
            )
            tiled_candidates = tf.tile(
                candidates[:, None, :], [1, vectorization, 1]
            )  # [samples, V, D]

        # Modif here
        # target_func_values = target_func(tiled_candidates)  # [samples, V]
        # debug = targetS_func(jnp.asarray(tiled_candidates.numpy())[0])
        target_func_values = tf.cast(
            vmap(target_func, in_axes=(0,))(jnp.asarray(tiled_candidates.numpy())),
            dtype=tiled_candidates.dtype,
        )
        target_func_values = tf.squeeze(target_func_values, axis=-2)
        # target_func_values = tf.cast(target_func(jnp.asarray(tiled_candidates.numpy())), dtype=tiled_candidates.dtype)  # [samples, V]

        tf.debugging.assert_shapes(
            [(target_func_values, ("_", vectorization))],
            message=(
                f"""
                The result of function target_func has shape
                {tf.shape(target_func_values)}, however, expected a trailing
                dimension of size {vectorization}.
                """
            ),
        )

        # now that we know the output dimension and dtypes, initialize empty top tensors
        if top_candidates is None:
            top_candidates = tf.zeros(
                [vectorization, 0, tf.shape(candidates)[-1]], dtype=candidates.dtype
            )
        if top_fun_values is None:
            top_fun_values = tf.zeros(
                [vectorization, 0], dtype=target_func_values.dtype
            )

        top_candidates = tf.concat(
            [top_candidates, tf.transpose(tiled_candidates, [1, 0, 2])], 1
        )  # [V, samples+num_initial_points, D]
        top_fun_values = tf.concat(
            [top_fun_values, tf.transpose(target_func_values)], 1
        )  # [V, samples+num_initial_points]

        _, top_k_indices = tf.math.top_k(
            top_fun_values, k=min(num_initial_points, tf.shape(top_fun_values)[-1])
        )  # [V, num_initial_points]

        top_candidates = tf.gather(
            top_candidates, top_k_indices, batch_dims=1
        )  # [V, num_initial_points, D]
        top_fun_values = tf.gather(
            top_fun_values, top_k_indices, batch_dims=1
        )  # [V, num_initial_points]

    if top_candidates is None:
        raise ValueError("No initial point generated!")

    initial_points = tf.transpose(top_candidates, [1, 0, 2])  # [num_initial_points,V,D]
    return initial_points


def _perform_parallel_continuous_optimization_jax_ver(
    target_func: AcquisitionFunction,
    space: SearchSpace,
    starting_points: TensorType,
    optimizer_args: dict[str, Any],
) -> Tuple[TensorType, TensorType, TensorType, TensorType]:
    """
    A function to perform parallel optimization of our acquisition functions
    using Scipy. We perform L-BFGS-B starting from each of the locations contained
    in `starting_points`, i.e. the number of individual optimization runs is
    given by the leading dimension of `starting_points`.

    To provide a parallel implementation of Scipy's L-BFGS-B that can leverage
    batch calculations with TensorFlow, this function uses the Greenlet package
    to run each individual optimization on micro-threads.

    L-BFGS-B updates for each individual optimization are performed by
    independent greenlets working with Numpy arrays, however, the evaluation
    of our acquisition function (and its gradients) is calculated in parallel
    (for each optimization step) using Tensorflow.

    For :class:'CollectionSearchSpace' we only apply gradient updates to
    its :class:'Box' subspaces, fixing the discrete elements to the best values
    found across the initial random search. To fix these discrete elements, we
    optimize over a continuous :class:'Box' relaxation of the discrete subspaces
    which has equal upper and lower bounds, i.e. we specify an equality constraint
    for this dimension in the scipy optimizer.

    This function also support the maximization of vectorized target functions (with
    vectorization V).

    :param target_func: The function(s) to maximise, with input shape [..., V, D] and
        output shape [..., V].
    :param space: The original search space.
    :param starting_points: The points at which to begin our optimizations of shape
        [num_optimization_runs, V, D]. The leading dimension of
        `starting_points` controls the number of individual optimization runs
        for each of the V target functions.
    :param optimizer_args: Keyword arguments to pass to the Scipy optimizer.
    :return: A tuple containing the failure statuses, maximum values, maximisers and
        number of evaluations for each of our optimizations.
    """

    tf_dtype = starting_points.dtype  # type for communication with Trieste

    num_optimization_runs_per_function = tf.shape(starting_points)[0].numpy()

    V = tf.shape(starting_points)[-2].numpy()  # vectorized batch size
    D = tf.shape(starting_points)[-1].numpy()  # search space dimension
    num_optimization_runs = num_optimization_runs_per_function * V

    vectorized_starting_points = tf.reshape(
        starting_points, [-1, D]
    )  # [num_optimization_runs*V, D]

    def _objective_value(vectorized_x: ArrayLike) -> ArrayLike:  # [N, D] -> [N, 1]
        vectorized_x = vectorized_x[
            None, ...
        ]  # [N, D] , this is used to be compatible with jax vmap which has reduced the batch axis
        # vectorized_x = vectorized_x[:, None, :]  # [N, 1, D] 
        # x = tf.reshape(vectorized_x, [-1, V, D])  # [N/V, V, D]
        # x = jnp.reshape(vectorized_x, [-1, V, D])  # [N/V, V, D]
        evals = -target_func(vectorized_x)  # [N/V, V]
        # evals = -target_func(x)  # [N/V, V]
        # vectorized_evals = tf.reshape(evals, [-1, 1])  # [N, 1]
        vectorized_evals = jnp.reshape(evals, [-1, 1])  # [N, 1]
        return np.squeeze(
            vectorized_evals
        )  # this squeeze is used to be compatible with jax vmap which needs output to be scalar

    # def _objective_value_and_gradient(x: TensorType) -> Tuple[TensorType, TensorType]:
    #     return tfp.math.value_and_gradient(_objective_value, x)  # [len(x), 1], [len(x), D]

    def _objective_value_and_gradient(x: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        return vmap(value_and_grad(_objective_value))(
            jnp.asarray(x)
        )  # [len(x), 1], [len(x), D]

    # it was observed from https://github.com/scipy/scipy/issues?q=_minimize_trustregion_constr+is%3Aclosed
    # that the trust-constr method is not stable as it may vioate the bound constraint, which is prohibited in our optimization cases
    # hence, follow the suggestion of https://github.com/scipy/scipy/pull/11712 and several related issues by using
    if isinstance(
        space, TaggedProductSearchSpace
    ):  # build continuous relaxation of discrete subspaces
        bounds = [
            get_bounds_of_box_relaxation_around_point(
                space, vectorized_starting_points[i : i + 1]
            )
            for i in tf.range(num_optimization_runs)
        ]
    elif isinstance(space, TaggedMultiSearchSpace):
        bounds = [
            spo.Bounds(lower, upper)
            for lower, upper in zip(space.subspace_lower, space.subspace_upper)
        ]
        # The bounds is a sequence of tensors, stack them into a single tensor. In this case
        # the vectorization of the target function must be a multple of the length of the sequence.
        remainder = V % len(bounds)
        tf.debugging.assert_equal(
            remainder,
            tf.cast(0, dtype=remainder.dtype),
            message=(
                f"""
                The vectorization of the target function {V} must be a multiple of the length
                of the bounds sequence {len(bounds)}.
                """
            ),
        )
        multiple = V // len(bounds)
        bounds = bounds * multiple * num_optimization_runs_per_function
    else:
        # i.e., we add keep_feasible=True here
        bounds = [
            spo.Bounds(space.lower, space.upper) # , keep_feasible=True
        ] * num_optimization_runs
    # print(bounds)
    # Initialize the numpy arrays to be passed to the greenlets
    np_batch_x = np.zeros(
        (num_optimization_runs, tf.shape(starting_points)[-1]), dtype=np.float64
    )
    np_batch_y = np.zeros((num_optimization_runs,), dtype=np.float64)
    np_batch_dy_dx = np.zeros(
        (num_optimization_runs, tf.shape(starting_points)[-1]), dtype=np.float64
    )

    # Set up child greenlets
    child_greenlets = [ScipyOptimizerGreenlet() for _ in range(num_optimization_runs)]
    vectorized_child_results: List[
        Union[spo.OptimizeResult, "np.ndarray[Any, Any]"]
    ] = [
        gr.switch(
            vectorized_starting_points[i].numpy(),
            bounds[i],
            space.constraints,
            optimizer_args,
        )
        for i, gr in enumerate(child_greenlets)
    ]

    _iter = 0
    while True:
        # print(f'acq_opt_iter: {_iter}')
        all_done = True
        for i, result in enumerate(
            vectorized_child_results
        ):  # Process results from children.
            if isinstance(result, spo.OptimizeResult):
                continue  # children return a `spo.OptimizeResult` if they are finished
            all_done = False
            assert isinstance(
                result, np.ndarray
            )  # or an `np.ndarray` with the query `x` otherwise
            np_batch_x[i, :] = result

        if all_done:
            break

        # Batch evaluate query `x`s from all children.
        # batch_x = tf.constant(np_batch_x, dtype=tf_dtype)  # [num_optimization_runs, d]
        if np.any(np_batch_x > bounds[0].ub):
            # print("Upper bound violated, force it back to the bound")
            np_batch_x = np.minimum(np_batch_x, bounds[0].ub)
        if np.any(np_batch_x < bounds[0].lb):
            # print("Lower bound violated, force it back to the bound")
            np_batch_x = np.maximum(np_batch_x, bounds[0].lb)
        # print(np_batch_x[0])
        batch_y, batch_dy_dx = _objective_value_and_gradient(np_batch_x)
        # np_batch_y = batch_y.numpy().astype("float64")
        np_batch_y = np.asarray(batch_y).astype("float64")
        # np_batch_dy_dx = batch_dy_dx.numpy().astype("float64")
        np_batch_dy_dx = np.asarray(batch_dy_dx).astype("float64")
        # print(f'acq_grad_norm: {np.linalg.norm(np_batch_dy_dx, axis=-1)}')

        for i, greenlet in enumerate(
            child_greenlets
        ):  # Feed `y` and `dy_dx` back to children.
            if greenlet.dead:  # Allow for crashed greenlets
                continue
            vectorized_child_results[i] = greenlet.switch(
                np_batch_y[i], np_batch_dy_dx[i, :]
            )

        _iter += 1

    final_vectorized_child_results: List[spo.OptimizeResult] = vectorized_child_results
    vectorized_successes = tf.constant(
        [result.success for result in final_vectorized_child_results]
    )  # [num_optimization_runs]
    vectorized_fun_values = tf.constant(
        [-result.fun for result in final_vectorized_child_results], dtype=tf_dtype
    )  # [num_optimization_runs]
    vectorized_chosen_x = tf.constant(
        [result.x for result in final_vectorized_child_results], dtype=tf_dtype
    )  # [num_optimization_runs, D]
    vectorized_nfev = tf.constant(
        [result.nfev for result in final_vectorized_child_results], dtype=tf_dtype
    )

    # Ensure chosen points satisfy any constraints in the search-space.
    if space.has_constraints:
        is_feasible = space.is_feasible(vectorized_chosen_x)
        vectorized_successes = tf.logical_and(vectorized_successes, is_feasible)

    successes = tf.reshape(vectorized_successes, [-1, V])  # [num_optimization_runs, V]
    fun_values = tf.reshape(
        vectorized_fun_values, [-1, V]
    )  # [num_optimization_runs, V]
    chosen_x = tf.reshape(
        vectorized_chosen_x, [-1, V, D]
    )  # [num_optimization_runs, V, D]
    nfev = tf.reshape(vectorized_nfev, [-1, V])  # [num_optimization_runs, V]

    return (successes, fun_values, chosen_x, nfev)


def init_cond_and_time_batchify_joint(
    batch_size_one_optimizer: AcquisitionOptimizer[SearchSpaceType],
    batch_size: int,
    state_dim: int
) -> AcquisitionOptimizer[SearchSpaceType]:
    """
    A wrapper around our :const:`AcquisitionOptimizer`s. This class wraps a
    :const:`AcquisitionOptimizer` to allow it to jointly optimize the batch elements considered
    by a batch acquisition function.

    :param batch_size_one_optimizer: An optimizer that returns only batch size one, i.e. produces a
            single point with shape [1, D].
    :param batch_size: The number of points in the batch.
    :return: An :const:`AcquisitionOptimizer` that will provide a batch of points with shape [B, D].
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    # t0 is explicitly needed as it does not have to be optimized in the batch
    def optimizer(
        expanded_search_space: SearchSpace,
        t0, # initial time
        f: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
        initial_cond: Optional[TensorType] = None,
    ) -> TensorType:

        if isinstance(f, tuple):
            raise ValueError(
                "batchify_joint cannot be applied to a vectorized acquisition function"
            )
        af: AcquisitionFunction = f  # type checking can get confused by closure of f

        def target_func_with_vectorized_inputs(
            x: TensorType,
        ) -> TensorType:  # [..., 1, B * D] -> [..., 1]
            """
            :params x: [..., 1, B * D]
            """
            
            if initial_cond is None:
                x0 = tf.repeat(x[..., :-batch_size], repeats=(batch_size + 1), axis=-2)
            else:
                x0 = tf.repeat(
                    tf.repeat(
                        initial_cond[None, None, ...], repeats=(batch_size + 1), axis=-2
                    ),
                    repeats=x.shape[0],
                    axis=-3,
                )
            ts = x[..., -batch_size:]
            t0_broadcasted = tf.cast( # expand t0 to the same shape as x0
                tf.broadcast_to(tf.expand_dims(t0, axis=-1), tf.shape(x0[..., :1, :1])),
                dtype=x.dtype,
            )
            ts = tf.reshape(ts, ts.shape[:-2].as_list() + [batch_size, -1]) 
            ts_with_t0 = tf.concat([t0_broadcasted, ts], axis=-2)
            x0_ts_with_t0 = tf.concat([x0, ts_with_t0], axis=-1)
            return af(x0_ts_with_t0)

        vectorized_points, val = batch_size_one_optimizer(  # [1, B * D]
            expanded_search_space, target_func_with_vectorized_inputs
        )
        if initial_cond is None:
            return tf.reshape(vectorized_points, [batch_size + state_dim, -1]), val  # [B, D]
        else:
            return tf.reshape(vectorized_points, [batch_size, -1]), val  # [B, D]

    return optimizer


def time_batchify_joint(
    batch_size_one_optimizer: AcquisitionOptimizer[SearchSpaceType],
    batch_size: int,
) -> AcquisitionOptimizer[SearchSpaceType]:
    """
    A wrapper around our :const:`AcquisitionOptimizer`s. This class wraps a
    :const:`AcquisitionOptimizer` to allow it to jointly optimize the batch elements considered
    by a batch acquisition function.

    :param batch_size_one_optimizer: An optimizer that returns only batch size one, i.e. produces a
            single point with shape [1, D].
    :param batch_size: The number of points in the batch.
    :return: An :const:`AcquisitionOptimizer` that will provide a batch of points with shape [B, D].
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    def optimizer(
        initial_cond,
        expanded_search_space: SearchSpace,
        f: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
    ) -> TensorType:

        if isinstance(f, tuple):
            raise ValueError(
                "batchify_joint cannot be applied to a vectorized acquisition function"
            )
        af: AcquisitionFunction = f  # type checking can get confused by closure of f

        def target_func_with_vectorized_inputs(
            x: TensorType,
        ) -> TensorType:  # [..., 1, B * D] -> [..., 1]
            x0 = tf.cast(
                tf.repeat(
                    tf.repeat(
                        initial_cond[None, None, ...], repeats=(batch_size), axis=-2
                    ),
                    repeats=x.shape[0],
                    axis=-3,
                ),
                dtype=x.dtype,
            )
            ts = tf.reshape(x, x.shape[:-2].as_list() + [batch_size, -1])
            x0_ts = tf.concat([x0, ts], axis=-1)
            return af(tf.cast(x0_ts, dtype=tf.float64))

        vectorized_points, val = batch_size_one_optimizer(  # [1, B * D]
            expanded_search_space, target_func_with_vectorized_inputs
        )
        return tf.reshape(vectorized_points, [batch_size, -1]), val  # [B, D]

    return optimizer


def init_cond_and_time_batchify_joint_jax_ver(
    batch_size_one_optimizer: AcquisitionOptimizer[SearchSpaceType],
    batch_size: int,
    state_dim: int
) -> AcquisitionOptimizer[SearchSpaceType]:
    """
    A wrapper around our :const:`AcquisitionOptimizer`s. This class wraps a
    :const:`AcquisitionOptimizer` to allow it to jointly optimize the batch elements considered
    by a batch acquisition function.

    :param batch_size_one_optimizer: An optimizer that returns only batch size one, i.e. produces a
            single point with shape [1, D].
    :param batch_size: The number of points in the batch.
    :return: An :const:`AcquisitionOptimizer` that will provide a batch of points with shape [B, D].
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    def optimizer(
        expanded_search_space: SearchSpace,
        t0,
        f: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
        initial_cond: Optional[ArrayLike] = None,
    ) -> TensorType:

        if isinstance(f, tuple):
            raise ValueError(
                "batchify_joint cannot be applied to a vectorized acquisition function"
            )
        af: AcquisitionFunction = f  # type checking can get confused by closure of f

        @jit
        def target_func_with_vectorized_inputs(
            x: ArrayLike,
        ) -> ArrayLike:  # [..., 1, B * D] -> [..., 1]
            """
            :params x: [..., 1, B * D]
            """
            if initial_cond is None:
                x0 = x[..., :-batch_size]
            else:
                x0 = jnp.repeat(initial_cond[None, ...], repeats=x.shape[0], axis=0)
            ts = x[..., -batch_size:]
            t0_broadcasted = jnp.broadcast_to(
                jnp.expand_dims(t0, axis=-1), jnp.shape(x0[..., :1, :1])
            )
            sorted_ts = jnp.sort(ts, axis=-1)
            sorted_ts_with_t0 = jnp.concatenate([t0_broadcasted, sorted_ts], axis=-1)
            x0_sorted_ts_with_t0 = jnp.concatenate([x0, sorted_ts_with_t0], axis=-1)
            return af(x0_sorted_ts_with_t0)

        vectorized_points, val = batch_size_one_optimizer(  # [1, B * D]
            expanded_search_space, target_func_with_vectorized_inputs
        )
        if initial_cond is None:
            return tf.reshape(vectorized_points, [batch_size + state_dim, -1]), val  # [B, D]
            # return tf.reshape(vectorized_points, [batch_size + 1, -1]), val  # [B, D]
        else:
            return tf.reshape(vectorized_points, [batch_size, -1]), val  # [B, D]

    return optimizer


def time_batchify_joint_jax_ver(
    batch_size_one_optimizer: AcquisitionOptimizer[SearchSpaceType],
    batch_size: int,
) -> AcquisitionOptimizer[SearchSpaceType]:
    """
    A wrapper around our :const:`AcquisitionOptimizer`s. This class wraps a
    :const:`AcquisitionOptimizer` to allow it to jointly optimize the batch elements considered
    by a batch acquisition function.

    :param batch_size_one_optimizer: An optimizer that returns only batch size one, i.e. produces a
            single point with shape [1, D].
    :param batch_size: The number of points in the batch.
    :return: An :const:`AcquisitionOptimizer` that will provide a batch of points with shape [B, D].
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    def optimizer(
        initial_cond,
        expanded_search_space: SearchSpace,
        f: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
    ) -> TensorType:

        if isinstance(f, tuple):
            raise ValueError(
                "batchify_joint cannot be applied to a vectorized acquisition function"
            )
        af: AcquisitionFunction = f  # type checking can get confused by closure of f

        @jit
        def target_func_with_vectorized_inputs(
            x: TensorType,
        ) -> TensorType:  # [..., 1, B * D] -> [..., 1]
            if initial_cond is None:
                x0 = x[..., :-batch_size]
                # x0 = jnp.repeat(x[..., :-batch_size], repeats=(batch_size + 1), axis=-2)
            else:
                # x0 = tf.repeat(tf.repeat(initial_cond[None, None, ...], repeats=(batch_size + 1), axis=-2), repeats=x.shape[0], axis=-3)
                x0 = jnp.repeat(
                    initial_cond[None, ...], repeats=x.shape[0], axis=0
                )  # jnp.repeat(jnp.repeat(initial_cond[None, None, ...], repeats=(batch_size + 1), axis=-2), repeats=x.shape[0], axis=-3)
            # x0 = tf.repeat(tf.repeat(initial_cond[None, None, ...], repeats=(batch_size), axis=-2), repeats=x.shape[0], axis=-3)

            ts = x[..., -batch_size:]
            sorted_ts = jnp.sort(ts, axis=-1)
            x0_sorted_ts_with_t0 = jnp.concatenate([x0, sorted_ts], axis=-1)
            return af(x0_sorted_ts_with_t0)
            # ts = tf.reshape(x, x.shape[:-2].as_list() + [batch_size, -1])
            # x0_ts = tf.concat([x0, ts], axis=-1)
            # return af(x0_ts)

        vectorized_points, val = batch_size_one_optimizer(  # [1, B * D]
            expanded_search_space, target_func_with_vectorized_inputs
        )
        # return tf.reshape(vectorized_points, [batch_size, -1]), val  # [B, D]
        if initial_cond is None:
            return tf.reshape(vectorized_points, [batch_size + 1, -1]), val  # [B, D]
        else:
            return tf.reshape(vectorized_points, [batch_size, -1]), val  # [B, D]

    return optimizer
