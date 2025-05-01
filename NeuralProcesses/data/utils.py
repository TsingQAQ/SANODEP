import tensorflow as tf
from jax import random
from jax import vmap
from jax import numpy as np
from jax import jit
from functools import partial
from jax.typing import ArrayLike
from typing import Optional
from jax import lax


def _cholesky(matrix):
    """Return a Cholesky factor and boolean success."""
    try:
        chol = tf.linalg.cholesky(matrix)
        ok = tf.reduce_all(tf.math.is_finite(chol))
        return chol, ok
    except tf.errors.InvalidArgumentError:
        return matrix, False


def safer_cholesky(matrix, max_attempts: int = 10, jitter: float = 1e-6):
    def update_diag(matrix, jitter):
        diag = tf.linalg.diag_part(matrix)
        diag_add = tf.ones_like(diag) * jitter
        new_diag = diag_add + diag
        new_matrix = tf.linalg.set_diag(matrix, new_diag)
        return new_matrix

    def cond(state):
        return state[0]

    def body(state):

        _, matrix, jitter, _ = state
        res, ok = _cholesky(matrix)
        new_matrix = tf.cond(ok, lambda: matrix, lambda: update_diag(matrix, jitter))
        break_flag = tf.logical_not(ok)
        return [(break_flag, new_matrix, jitter * 10, res)]

    jitter = tf.cast(jitter, matrix.dtype)
    init_state = (True, update_diag(matrix, jitter), jitter, matrix)
    result = tf.while_loop(cond, body, [init_state], maximum_iterations=max_attempts)

    return result[-1][-1]


def context_target_split(rng: random.PRNGKey, x, y, num_context, num_extra_target):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.
    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)
    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)
    num_context : int
        Number of context points.
    num_extra_target : int
        Number of additional target points.
    """
    num_points = x.shape[1]
    # Sample locations of context and target points
    locations = random.choice(
        key=rng,
        a=np.arange(num_points),
        shape=(num_context + num_extra_target,),
        replace=False,
    )
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations, :]
    y_target = y[:, locations, :]
    return x_context, y_context, x_target, y_target


@partial(
    jit, static_argnames=("batch_size", "num_timesteps", "problem_setting_forcasting_prob", "know_initial")
)
def context_target_mask_gen_dynamic_system(
    rng: random.PRNGKey,
    batch_size: int,
    known_traj: ArrayLike,
    num_timesteps: ArrayLike,
    num_context: ArrayLike,
    num_extra_target: ArrayLike,
    problem_setting_forcasting_prob: float,
    know_initial: bool = False,
):
    """
    Given a batch of data each with num_points length, randomly split each row to context and target part with
    specified context numbers and target numbers. Note that since we want to infer the dynamics from multiple
    trajectories, here our data has been added a new dimensionality representing the trajectory_size dimensinality
    as the number of trajectories in the batch

    this mask used to resemble the following senarios:
      if we have observed some trajectories within a batch, we may be able to infer its state at some certain time
      point, but we should also be able to infer some whole new trajectories starting at different initial conditions

    :params use_initial, whether to use the starting point (the first point), this is used when dealing with temperal
        related series like Nueral ODE dataset when the initial condition is always known
    :params batch_size
    :param known_traj: [batch_size] how many trajectories do we know in each batch, this is used to emulate the varing
        trajectories, we have observed in reaction process
    :param num_points: int the total number of points (measurements) in the batch
    :param num_context: [batch_size, traj_size] the number of context points, for implementation convenicne, it can be different
        across different data in a trajectory but will be the same across different data batch, note that num_context must be at least 1
        otherwise there will be issues in generate_locations when use_initial is set to True due to num_context - use_initial without
        a warning!
    :param num_extra_target: [batch_size, traj_size] the number of extra target points, for implementation convenicne, it can be different
        across different data in a trajectory but will be the same across different data batch

    :reurn 
        context_mask [batch_size, traj_size, num_points]
        target_mask [batch_size, traj_size, num_points]
        context_mask_existing_known_traj [batch_size, traj_size, num_points]
        hybrid_formulation_mask [batch_size, traj_size, traj_size, num_points]
        context_mask_with_new_traj_target_mask  [batch_size, traj_size, traj_size, num_points]
        target_initial_cond_mask 
        target_mask_unknown_traj,
    """

    # Sample locations of context and target points
    num_target = (
        num_context + num_extra_target
    )  # [batch_size, traj_size], calculate the timestep sizesX
    num_system, traj_size = num_context.shape[0], num_context.shape[-1]
    rng, bernoulli_rng = random.split(rng)
    rng_array = random.split(rng, (batch_size, traj_size))

    def generate_location_mask(key: random.PRNGKey, size: int, num_context: int, num_target: int, known_initial_cond: Optional[bool]=False) -> tuple[ArrayLike]:
        """
        according to the given context size and target size, randomly generate the context and target mask that matches the size, note that 
        context mask (of True) is a subset of target mask, 
        
        note to be compatible with vmap: it will first generate a random shuffled array which consists of the index of all points 
        (excluding the initial condition if it is known), then it specified two thresholds where the number in array below these two 
        thresholds will corresponds to context and target number respectively. Then the boolean masks are generated by comparing the 
        array with the thresholds, this has garantted that the context mask is a subset of target mask

        :params size total number of points
        :params num_context number of context points
        :params num_target  number of target points
        :param use_initial
        """
        # FIXME: When num_context is 0 and use_initial is True, there will be issues, 
        # since we do not expect any num_context to be zero, we ignore this issue for now
        num_context = num_context - known_initial_cond
        num_target = num_target - known_initial_cond

        # Generate a random array
        array = random.permutation(
            key, np.arange(size - known_initial_cond)
        )  # random.uniform(key, (size - use_initial,))
        _helper_sorted_array = np.sort(array)
        
        # when the original context number is 1 and known_initial_cond is True, 
        # num_context will be 0, so we need to set the threshold to -1 (i.e., arbitary value that is less than any value in the array)
        ctx_thresholds = lax.cond(
            np.any(num_context == 0),
            lambda _: -1,
            lambda _: _helper_sorted_array[num_context - 1],
            operand=None, # operand=None indicates that the condition function does not have any additional input
        )

        # same as above
        tgt_thresholds = lax.cond(
            np.any(num_target == 0),
            lambda _: -1,
            lambda _: _helper_sorted_array[num_target - 1],
            operand=None,
        )

        # Create masks by comparing the array with the thresholds
        context_locations = array <= ctx_thresholds
        target_locations = array <= tgt_thresholds

        if known_initial_cond:
            initial = np.ones(shape=(1,)).astype(np.bool_)
        else:
            initial = np.ones(shape=(0,)).astype(np.bool_)
        context_locations = np.concatenate([initial, context_locations])
        target_locations = np.concatenate([initial, target_locations])

        return context_locations, target_locations


    # context_locations, target_locations = generate_location_mask(rng_array[1, 1], num_timesteps, num_context[1, 1], num_target[1, 1], know_initial)
    # generate context mask and target mask for all trajectories in the system sample, note that this function do not take known_traj within into account
    context_locations, target_locations = vmap(
        vmap(generate_location_mask, in_axes=(0, None, 0, 0, None)),
        in_axes=(0, None, 0, 0, None),
    )(rng_array, num_timesteps, num_context, num_target, know_initial) 

    # additional masking for the known trajectories: note that for this part, unlike time, since each trajectory do not have a order between each other
    # we take a simple approach by using the first known_traj trajectories as the known trajectories 

    # Create a helper range array of shape [batch_size, traj_size]
    helper_range_array = np.arange(traj_size)[None, :] * np.ones(
        (batch_size, 1)
    )  # [batch_size, traj_size]

    # Create a known_traj array of shape [batch_size, traj_size]
    known_traj_array = known_traj[:, None] * np.ones(
        (1, traj_size)
    )  # [batch_size, traj_size]

    # Create the init_cond_mask
    known_traj_mask = helper_range_array < known_traj_array  # [batch_size, traj_size]

    # expand the known_traj_mask to include the additional dimension to take into accounte the timestep wise mask
    known_traj_mask = np.repeat(
        known_traj_mask[..., None], num_timesteps, axis=-1
    )  # [batch_size, traj_size, num_points]
    context_mask_existing_known_traj = np.logical_and(
        context_locations, known_traj_mask
    )  # [batch_size, traj_size, num_points]

    # ====================================================================================================

    # Problem formulation 1: 
    # in order to meta learn a whole dynamic system for optimization, problem formulation 1 considers a forecasting problem: 
    # given a dynamic system with some known trajectories, we want to predict the new trajectories starting at initial condition
    # by only know its initial condition state value and the other known trajectoires

    # in order to do so, we need to augmented our context mask with the new trajectories initial condition mask, we achive this point by 
    # using a mini batch way: since we have a known traj_size, and within there will only be `known_traj` number of known trajectories, 
    # we sqeuqntially treat each of the trajectory as a new trajectory and use the rest of the known trajectories, together with this 
    # new trajectore's initial condition as the context data, this basically means we perform a Monte Carlo sampling of x0 using 
    # traj_size MC numbers (ignoiring that some MC sampled x0 is the same as the known x0). To the end, we will have a batch of 
    # augmented new initial condition's context mask wih the following shape
    # [batch_size, traj_size, traj_size, num_points]

    # The current implementation of forecasting have one potential concern, that is due to the logical_or operatio, for the first known_traj 
    # trajectories, it actually perform forecasting on the same trajectory it is known (hence interpolating), which means 
    # known_traj / traj_size will actually be conducting interpolating.

    # [batch_size, traj_size, traj_size]
    additional_new_traj_init_cond_ctx_mask = np.repeat(np.eye(traj_size)[None, ...], num_system, axis=0)
    
    # fill the additional timestep axis: add all the rest of the timesteps to be False except the initial condition
    additional_new_context_init_cond_mask = np.pad(
    additional_new_traj_init_cond_ctx_mask[..., None],
    pad_width=((0, 0), (0, 0), (0, 0), (0, num_timesteps - 1)), # add num_timesteps - 1 zeros at the end of last axis
    mode='constant',
    constant_values=False
    ).astype(np.bool_)

    context_mask_with_new_traj_init_cond = np.logical_or(
        additional_new_context_init_cond_mask,  # this is correct
        np.repeat(
            np.expand_dims(context_mask_existing_known_traj, -3), traj_size, axis=-3
        ),
    )

    # Problem formulation 2 # [batch_size, traj_size, traj_size, num_points]
    # problem formulation 2 considers a interpolating problem: 
    # given a dynamic system with some known trajectories, and we have started with a new trajectory and 
    # observed some state values at differen timesteps, we want to predict along these new trajectory at different time

    helper_aug_axis_init_cond_masks = np.repeat(
        additional_new_traj_init_cond_ctx_mask[..., None], num_timesteps, axis=-1
    )  # [batch_size, traj_size, traj_size, num_points]
    # create the original context with the new trajectory's observations as the new context mask

    additional_context_all_traj_mask = np.logical_or(
        helper_aug_axis_init_cond_masks,
        np.repeat(np.expand_dims(known_traj_mask, -3), traj_size, axis=-3),
    )
    context_with_new_traj_obs_mask = np.logical_and(
        np.repeat(np.expand_dims(context_locations, -3), traj_size, axis=-3),
        additional_context_all_traj_mask,
    )

    # create the original context with the new trajectory's target observations as the new target mask
    additional_new_traj_target_mask = np.logical_and(
        helper_aug_axis_init_cond_masks,
        np.repeat(np.expand_dims(target_locations, -3), traj_size, axis=-3),
    )

    context_mask_with_new_traj_target_mask = np.logical_or(
        additional_new_traj_target_mask,
        np.repeat(
            np.expand_dims(context_mask_existing_known_traj, -3), traj_size, axis=-3
        ),
    )
    # only initial condition target mask
    context_mask = context_locations
    target_mask = target_locations

    target_initial_cond_mask = np.concatenate(
        [
            np.ones(shape=(batch_size, traj_size, 1)),
            np.zeros(shape=(batch_size, traj_size, num_timesteps - 1)),
        ],
        axis=-1,
    )

    # Use np.all to create a mask for the num_points dimension
    known_traj_mask = np.repeat(
        (helper_range_array >= known_traj_array)[..., None], num_timesteps, axis=-1
    )  # [batch_size, traj_size, num_points]
    target_mask_unknown_traj = np.logical_and(
        target_mask, known_traj_mask
    )  # [batch_size, traj_size, num_points]
    
    # for the optimization problem, in a same dynamic system, we need to consider both the forcasting problem and the regression problem
    # hence we hybrid the mask using a bernoulli distribution, with the probability of problem_setting_forcasting_prob
    # given the problem_setting_forcasting_prob, we randomly combine context_mask_with_new_traj_init_cond
    forcasting_mask = random.bernoulli(bernoulli_rng, p=problem_setting_forcasting_prob, shape=(batch_size, traj_size)) # [batch_size, traj_size]
    expanded_forcasting_mask = np.repeat(np.repeat(forcasting_mask[..., None], traj_size, axis=-1)[..., None], num_timesteps, -1) # [batch_size, traj_size, num_points]
    hybrid_formulation_mask = expanded_forcasting_mask * context_mask_with_new_traj_init_cond +\
          (~expanded_forcasting_mask) * context_with_new_traj_obs_mask

    return (
        context_mask,
        target_mask,
        context_mask_existing_known_traj,
        hybrid_formulation_mask,
        context_mask_with_new_traj_target_mask,
        target_initial_cond_mask,
        target_mask_unknown_traj,
    )
