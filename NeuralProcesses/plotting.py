import logging
import os

import jax
import tensorflow as tf
from absl import flags
from flax.training import checkpoints, train_state
from jax import numpy as np
from ml_collections import ConfigDict
from tqdm import tqdm

from .data import datasets
from .models import build
from .train.optimizer import create_optimizer

FLAGS = flags.FLAGS


def plot_meta_learn_model(config: ConfigDict, workdir: str):
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    assert isinstance(config.seed, int)  # type check
    rng = jax.random.PRNGKey(config.seed)
    if config.model.name == "GP":
        model = None
        need_train_epoch = 0
        initial_step = 0
        trained_epoch = 0
        training_state = None
    else:
        rng, init_rng = jax.random.split(rng)
        model, init_model_state, initial_params, rng = build.init_model(
            init_rng, config
        )
        # instantiate optimizers

        # create training state
        if "create_optimizer" in config.training:
            optimizer = config.training["create_optimizer"](config)
        else:
            optimizer = create_optimizer(config)

        training_state = train_state.TrainState.create(
            apply_fn=model.apply, params=initial_params, tx=optimizer
        )
        # Create checkpoints directory
        checkpoint_dir = os.path.join(workdir, "checkpoints")

        # Resume training when intermediate checkpoints are detected
        training_state = checkpoints.restore_checkpoint(checkpoint_dir, training_state)
        # `state.step` is JAX integer on the GPU/TPU devices
        initial_step = int(training_state.step)
        trained_epoch = (
            initial_step // config.training.optimizer.args.num_steps_per_epoch
        )
        need_train_epoch = config.training.num_epochs - trained_epoch

    # Build data iterators
    rng, shuffle_rng = jax.random.split(rng)
    data, dataset_inst = datasets.get_dataset(config)
    # the order of shuffle, batch and repeat shall follows strictly as states here: https://stackoverflow.com/questions/49915925/output-differences-when-changing-order-of-batch-shuffle-and-repeat#:~:text=Best%20Ordering%3A&text=For%20batches%20to%20be%20different,are%20unique%2C%20unlike%20the%20other.
    if (
        config.data.args.generator is True
    ):  # if dataset is gen from generator, no need to shuffle and batch (assumed it has been done in dataset level)
        pass
    else:
        data = (
            data.shuffle(
                buffer_size=config.data.shuffle_buffer_size,
                seed=int(shuffle_rng[0]),
                reshuffle_each_iteration=True,
            )
            .batch(batch_size=config.training.batch_size)
            .repeat(need_train_epoch)
        )

    # auxilary data to supervise the model performance during training
    if config.data.args.get("aux"):
        aux_datsets = datasets.get_aux_datasets(dataset_inst, config)

    logging.info("Starting training loop at step %d." % (initial_step,))

    this_sample_dir = sample_dir  # os.path.join(, "epoch_{}".format(current_epoch))
    tf.io.gfile.makedirs(this_sample_dir)
    samples = config.sampling_fn(
        config=config,
        model=model,
        rng=rng,
        training_state=training_state,
        dataset_inst=dataset_inst,
        current_epoch=trained_epoch,
        aux_batch=aux_datsets,
        this_sample_dir=this_sample_dir,
    )




def plot_gp_model(config: ConfigDict, workdir: str):
    """
    Construct and evaluate the GP model

    Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    import csv

    import gpflow
    from tensorflow_probability import distributions as tfd
    from trieste.data import Dataset as trieste_Dataset
    from trieste.models.interfaces import \
        TrainablePredictJointReparamModelStack

    from NeuralProcesses.models.gpflow.models import (
        Safer_GPR, SaferGaussianProcessRegression)

    # As a first step, we construct a single output Gaussian Processes
    def build_model(data: trieste_Dataset):
        variance = 1.0
        kernel = gpflow.kernels.Matern32(
            variance=variance, lengthscales=[1.0] * data.query_points.shape[-1]
        )
        gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)

        return SaferGaussianProcessRegression(gpr, num_kernel_samples=100)

    def build_stacked_independent_objectives_model(
        data: trieste_Dataset, _num_states: int
    ) -> TrainablePredictJointReparamModelStack:
        gprs = []
        for idx in range(_num_states):
            single_state_data = trieste_Dataset(
                data.query_points, tf.gather(data.observations, [idx], axis=1)
            )
            gpr = build_model(single_state_data)
            gprs.append((gpr, 1))

        return TrainablePredictJointReparamModelStack(*gprs)

    # Build data iterators
    rng = jax.random.PRNGKey(config.seed)
    rng, shuffle_rng = jax.random.split(rng)
    data, dataset_inst = datasets.get_dataset(config)
    # the order of shuffle, batch and repeat shall follows strictly as states here: https://stackoverflow.com/questions/49915925/output-differences-when-changing-order-of-batch-shuffle-and-repeat#:~:text=Best%20Ordering%3A&text=For%20batches%20to%20be%20different,are%20unique%2C%20unlike%20the%20other.

    # Create data normalizer and its inverse
    pre_processor = datasets.get_data_preprocessor(dataset_inst, config)

    data_iterator = data.as_numpy_iterator()

    pbar = tqdm(
        data_iterator,
        total=config.evaluation.num_steps,
        initial=0,
        desc=f"Evaluation Step",
    )  # use int to convert jitted step (array) back that is acceptable for tqdm
    for iter, data_batch in enumerate(pbar):
        if iter == config.evaluation.num_steps:
            break
        processed_data, rng = pre_processor(
            data_batch,
            rng,
            known_traj_range=config.data.known_traj_range,
            all_as_target=False,
        )

        (
            data_t,
            data_x,
            _,
            _,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
            _,
            _,
            _,
        ) = processed_data
        # for computational issues, we only evaluate the first dynamic
        ctx_mask_with_new_traj_obs = ctx_mask_with_new_traj_obs[:1, :10, :10]
        ctx_mask_with_new_traj_target_mask = ctx_mask_with_new_traj_target_mask[
            :1, :10, :10
        ]
        data_t = data_t[:1, :10]  #
        data_x = data_x[:1, :10]

        # maybe we do not want to do this whole big loop!
        num_states = data_x.shape[-1]

        metrics = {"mse": [], "nll": []}
        for (
            data_t_single_dynamics,
            data_x_single_dynamics,
            ctx_mask_with_new_traj_obs_single_dynamic,
            ctx_mask_with_new_traj_target_mask_single_dynamic,
        ) in zip(
            data_t,
            data_x,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
        ):

            for traj_idx, (
                ctx_mask_with_new_traj_obs_single_traj,
                ctx_mask_with_new_traj_target_mask_single_traj,
            ) in tqdm(
                enumerate(
                    zip(
                        ctx_mask_with_new_traj_obs_single_dynamic,
                        ctx_mask_with_new_traj_target_mask_single_dynamic,
                    )
                )
            ):
                # training data
                augmented_input_init_cond = np.repeat(
                    data_x_single_dynamics[:, :1, :],
                    axis=1,
                    repeats=data_x_single_dynamics.shape[1],
                )
                augmented_input = np.concatenate(
                    [data_t_single_dynamics, augmented_input_init_cond], axis=-1
                )
                dataset = trieste_Dataset(
                    tf.cast(
                        augmented_input[ctx_mask_with_new_traj_obs_single_traj],
                        dtype=tf.float64,
                    ),
                    tf.cast(
                        data_x_single_dynamics[ctx_mask_with_new_traj_obs_single_traj],
                        dtype=tf.float64,
                    ),
                )

                # train the GP model

                models = build_stacked_independent_objectives_model(
                    dataset, _num_states=num_states
                )
                # The gp can fail to optimize
                models.update(dataset)
                models.optimize(dataset)

                # prediction
                test_dataset = trieste_Dataset(
                    tf.cast(
                        augmented_input[traj_idx][
                            ctx_mask_with_new_traj_target_mask_single_traj[traj_idx]
                        ],
                        dtype=tf.float64,
                    ),
                    tf.cast(
                        data_x_single_dynamics[traj_idx][
                            ctx_mask_with_new_traj_target_mask_single_traj[traj_idx]
                        ],
                        dtype=tf.float64,
                    ),
                )

                # visual checked correct
                # from matplotlib import pyplot as plt
                #
                # plt.figure()
                # plt.scatter(dataset.query_points[..., 0].numpy(), dataset.query_points[..., 1].numpy(), marker='x', s=50, zorder=100)
                # plt.scatter(test_dataset.query_points[..., 0].numpy(), test_dataset.query_points[..., 1].numpy())
                # plt.xlabel('Time')
                # plt.ylabel('Initial_Cond')
                # plt.savefig('training_data.png')

                gp_mean, gp_var = models.predict(test_dataset.query_points)
                # calculate MSE and negative log likelihood
                mse = ((gp_mean - test_dataset.observations) ** 2).numpy().mean()
                nll = (
                    -tfd.Normal(gp_mean, tf.sqrt(gp_var))
                    .log_prob(test_dataset.observations)
                    .numpy()
                    .mean()
                )
                metrics["mse"].append(mse)
                metrics["nll"].append(nll)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.join(workdir, "evaluations"), exist_ok=True)

    # Open the file in write mode
    with open(
        os.path.join(
            workdir,
            "evaluations",
            f"evaluation_metrics_mse_forcst_prob_{config.data.foracsting_problem_prob}.csv",
        ),
        "w",
    ) as f:
        # Create a CSV writer
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean", "Std"])
        writer.writerow(
            ["MSE", np.asarray(metrics["mse"]).mean(), np.asarray(metrics["mse"]).std()]
        )
        writer.writerow(
            ["NLL", np.asarray(metrics["nll"]).mean(), np.asarray(metrics["nll"]).std()]
        )
    mse_values = np.asarray(metrics["mse"])
    nll_values = np.asarray(metrics["nll"])

    # Save to .npz file
    np.savez(
        os.path.join(workdir, "evaluations", "metrics.npz"),
        mse=mse_values,
        nll=nll_values,
    )
