"""
This is where the experiment has been conducted
"""

import logging
import os

import jax
import pandas as pd
import tensorflow as tf
from absl import flags
from flax.metrics import tensorboard
from flax.training import checkpoints, train_state
from jax import numpy as np
from ml_collections import ConfigDict
from tqdm import tqdm

from .data import datasets
from .models import build
from .train.losses import get_train_eval_step_fn
from .train.optimizer import create_optimizer
from .utils.records import initialize_metrics_record, update_metrics

FLAGS = flags.FLAGS


def train_meta_learn_model(config: ConfigDict, workdir: str):
    """
    Runs the training pipeline.

    note that since the meta training data is generated on the fly and we currently do not save the training data 
    generating state in the checkpoint (where we always start with `config.seed`), this means that the training 
    data will be regenerated from beginning when the training is resumed, hence we do advice resuming training 
    at some checkpoint for reproducibility consideration, unless the state of training data generation is saved (TODO).

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    assert isinstance(config.seed, int)  # type check
    init_rng = jax.random.PRNGKey(config.seed)
    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    model, initial_params, _ = build.init_model(init_rng, config)
    
    # instantiate optimizers
    if "create_optimizer" in config.training:
        optimizer = config.training["create_optimizer"](config)
    else:
        optimizer = create_optimizer(config)

    # create training state
    training_state = train_state.TrainState.create(
        apply_fn=model.apply, params=initial_params, tx=optimizer
    )

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Resume training when intermediate checkpoints are detected
    training_state = checkpoints.restore_checkpoint(checkpoint_dir, training_state)
    # transform `state.step` from JAX integer (on the GPU/TPU devices) to int
    initial_trained_step = int(training_state.step)
    trained_epoch = initial_trained_step // config.training.optimizer.args.num_steps_per_epoch
    need_train_epoch = config.training.num_epochs - trained_epoch

    if initial_trained_step != 0:
        # raise the training data concern
        logging.warning(
            "The training data will be regenerated from beginning when the training is resumed, \
            hence we do advice resuming training at some checkpoint for reproducibility consideration."
        )

    # Build data iterators
    data_rng, shuffle_rng = jax.random.split(config.data.args.data_gen_rng, 2)
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

    # Create data preprocessor: currently it is mainly used to provide masking mechanism 
    # so we do not use the post_processor 
    pre_processor = datasets.get_data_preprocessor(dataset_inst, config)
    # NOTE be aware that the post_processor is not used atm
    # post_processor = datasets.get_data_post_processor(dataset_inst, config)

    # prepare one-step training and evaluation functions
    train_step_fn = get_train_eval_step_fn(
        model,
        training=True,
        optimize_fn=optimizer,
        loss_method=config.loss_method,
        config=config,
    )

    # auxilary data to supervise the model performance during training
    if config.data.args.get("aux"):
        aux_datsets = datasets.get_aux_datasets(dataset_inst, config)
        # auxilary dataset can be evaluated on different metrics
        eval_step_fn_collections = {
            key: get_train_eval_step_fn(
                model,
                training=False,
                optimize_fn=None,
                loss_method=metric,
                config=config,
            )
            for key, metric in config.data.aux_eval_metric.items()
        }

    logging.info("Starting training loop at step %d." % (initial_trained_step,))

    data_iterator = data.as_numpy_iterator()
    metric_record = initialize_metrics_record(config)
    last_epoch_metrics = metric_record.average # average across all steps & metric dimenions to a scalar value

    # training loop starts
    for current_epoch in np.arange(trained_epoch, config.training.num_epochs, 1):
        # initial_trained_step
        pbar = tqdm(
            data_iterator,
            total=config.training.optimizer.args.num_steps_per_epoch,
            initial=initial_trained_step # how many epochs have been trained
            % config.training.optimizer.args.num_steps_per_epoch,
            desc=f"Training Epoch: {current_epoch}",
        )  # use int to convert jitted step (array) back that is acceptable for tqdm

        for data_batch in pbar:
            processed_data, data_rng = pre_processor(
                data_batch, data_rng, known_traj_range=config.data.known_traj_range
            )

            (data_rng, training_state), loss, aux_info = train_step_fn((data_rng, training_state), processed_data)

            # evaluate auxilary data
            if config.data.args.get("aux"):
                aux_data_cfg = config.data.args.aux
                aux_metric = {}
                for key in aux_data_cfg.keys():
                    processed_aux_data, data_rng = pre_processor(
                        aux_datsets[key],
                        data_rng,
                        known_traj_range=config.data.known_traj_range,
                        all_as_target=True,
                    )
                    _, specific_aux_metric, _ = eval_step_fn_collections[
                        key
                    ]((data_rng, training_state), processed_aux_data)
                    aux_metric[key] = specific_aux_metric
            else:
                aux_metric = {}
            metric_record = update_metrics(new_metrics= aux_info |{"tr_loss": loss}|aux_metric, metric_record= metric_record)

            for metric, val in metric_record.average.items():
                writer.scalar(metric, val, training_state.step)  # tensorboard
            # writer.scaler('avg_ode_steps', np.mean(ode_steps), training_state.step)

            pbar.set_postfix(last_epoch_metrics)  # add metric description
            if (
                training_state.step % config.training.optimizer.args.num_steps_per_epoch
            ) == 0 and training_state.step != 0:  # reached a whole epoch
                break


        # summarized the records within epoch and reinitilize last epoch metric for next epoch
        last_epoch_metrics = metric_record.average
        metric_record = initialize_metrics_record(config)

        logging.info("epoch: %d, training_loss: %.5e" % (current_epoch, loss))
        # Save a checkpoint periodically and generate samples if needed
        if (
            current_epoch % config.training.snapshot_ckpt_freq == 0
        ):  # Save the checkpoint.
            checkpoints.save_checkpoint(
                checkpoint_dir, training_state, step=current_epoch, keep=np.inf
            )  # np.inf: keep every checkpoint

        # generate samples
        if (
            config.training.eval_snapshot_sampling is True
            and current_epoch != 0
            and current_epoch % config.training.snapshot_sampling_freq == 0
        ):
            this_sample_dir = (
                sample_dir  
            )
            tf.io.gfile.makedirs(this_sample_dir)
            config.sampling_fn(
                config=config,
                model=model,
                rng=config.snap_shot_sampling_cfg.sampling_rng,
                training_state=training_state,
                data_batch=data_batch,
                dataset_inst=dataset_inst,
                aux_batch=aux_datsets,
                this_sample_dir=this_sample_dir,
                current_epoch=current_epoch,
            )



def eval_meta_learn_model(config: ConfigDict, workdir: str):
    """
    Evaluate the traned meta learned model on a large number of test data
    
    Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    assert isinstance(config.seed, int)  # type check
    rng = config.evaluation.rng

    rng, init_rng = jax.random.split(rng)
    model, initial_params, rng = build.init_model(init_rng, config)
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

    # Build data iterators
    rng, shuffle_rng = jax.random.split(rng)
    data, dataset_inst = datasets.get_dataset(config)
    # the order of shuffle, batch and repeat shall follows strictly as states here: https://stackoverflow.com/questions/49915925/output-differences-when-changing-order-of-batch-shuffle-and-repeat#:~:text=Best%20Ordering%3A&text=For%20batches%20to%20be%20different,are%20unique%2C%20unlike%20the%20other.

    # Create data normalizer and its inverse
    pre_processor = datasets.get_data_preprocessor(dataset_inst, config)
    # NOTE be aware that the post_processor is not used atm
    # post_processor = datasets.get_data_post_processor(dataset_inst, config)

    eval_step_fn_collections = {
        key: get_train_eval_step_fn(
            model,
            training=False,
            optimize_fn=None,
            loss_method=metric,
            config=config,
        )
        for key, metric in config.data.eval_metrics.items()
    }

    data_iterator = data.as_numpy_iterator()
    metric_record = initialize_metrics_record(config)

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
            all_as_target=True,
        )

        metrics = {}
        for key in config.data.eval_metrics.keys():
            _, metric, _ = eval_step_fn_collections[key](
                (rng, training_state), processed_data
            )
            metrics[key] = metric
        metric_record.append(metrics)
    # save the evaluation metrics to a csv file
    # Construct the directory path
    workdir = os.path.join(
        workdir, "evaluations", f"ctx_traj_size_{config.data.known_traj_range}"
    )

    # Ensure the directory exists
    os.makedirs(workdir, exist_ok=True)

    # Open the file in write mode
    df = pd.DataFrame(metric_record.dict)
    df.to_csv(os.path.join(
            workdir,
            f"evaluation_metrics_forcst_prob_{config.data.foracsting_problem_prob}.csv",
        ), index=True)


def eval_gp_model(config: ConfigDict, workdir: str, sys_id: int = 0):
    """
    Construct and evaluate the GP model


    This can be a pure CPU based function hence we add the process_id to hopefully parallize and speed up the evaluation
    Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    sys_id: int, which system index within the batch to evaluate, this is used for parallel evaluation
    """

    from tensorflow_probability import distributions as tfd
    from trieste.data import Dataset as trieste_Dataset

    from NeuralProcesses.models.gpflow.builder import \
        build_stacked_independent_objectives_model

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
            all_as_target=True,
        )

        (
            data_t,
            data_x,
            _,
            _,
            _,
            ctx_mask_with_new_traj_obs,
            ctx_mask_with_new_traj_target_mask,
            _,
            _,
            _,
        ) = processed_data

        ctx_mask_with_new_traj_obs = ctx_mask_with_new_traj_obs[sys_id] # [num_traj, num_traj, timesteps]
        ctx_mask_with_new_traj_target_mask = ctx_mask_with_new_traj_target_mask[sys_id] # [num_traj, num_traj, timesteps]
        data_t = data_t[sys_id]
        data_x = data_x[sys_id]

        # maybe we do not want to do this whole big loop!
        num_states = data_x.shape[-1]

        metrics = {"mse": [], "nll": []}

        # loop through trajectories to focus
        for traj_idx, (
            ctx_mask_with_new_traj_obs_single_traj,
            ctx_mask_with_new_traj_target_mask_single_traj,
        ) in tqdm(
            enumerate(
                zip(
                    ctx_mask_with_new_traj_obs,
                    np.diagonal(ctx_mask_with_new_traj_target_mask, axis1=-3, axis2=-2),
                )
            )
        ):
            # training data
            augmented_input_init_cond = np.repeat(
                data_x[:, :1, :], # note thia :1 means the initial condition
                axis=1,
                repeats=data_x.shape[1],
            )
            augmented_input = np.concatenate(
                [data_t, augmented_input_init_cond], axis=-1
            )
            dataset = trieste_Dataset(
                tf.cast(
                    augmented_input[ctx_mask_with_new_traj_obs_single_traj],
                    dtype=tf.float64,
                ),
                tf.cast(
                    data_x[ctx_mask_with_new_traj_obs_single_traj],
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
            test_dataset = trieste_Dataset(
                tf.cast(
                    augmented_input[traj_idx][
                        ctx_mask_with_new_traj_target_mask_single_traj
                    ],
                    dtype=tf.float64,
                ),
                tf.cast(
                    data_x[traj_idx][
                        ctx_mask_with_new_traj_target_mask_single_traj
                    ],
                    dtype=tf.float64,
                ),
            )

            gp_mean, gp_var = models.predict(test_dataset.query_points)
            
            # calculate MSE and negative log likelihood
            # [timesteps, state_dim] -> 1 through averaging across state_dim and time_dim
            mse = ((gp_mean - test_dataset.observations) ** 2).numpy().mean()

            # we note that echo how the NLL is calculated in meta learning, we calculate the NLL here across trajectory as well
            # which is the summation across state and time dimensions
            nll = np.sum(
                -tfd.Normal(gp_mean, tf.sqrt(gp_var))
                .log_prob(test_dataset.observations)
                .numpy(),
                axis=[-1, -2],
            ) # [time_steps, state_dim] -> 1
            if np.all(np.isfinite(mse)):
                metrics["mse"].append(mse)
            if np.all(np.isfinite(nll)):
                metrics["nll"].append(nll)
            # print(metrics['mse'])
            # print(metrics['nll'])
            # print(np.asarray(metrics['nll']))
            # print(np.asarray(metrics['mse']))
            # print(
            #     f"mse: {mse}, nll: {nll}, mse_mean: {np.mean(np.asarray(metrics['mse']))}, nll_mean: {np.mean(np.asarray(metrics['nll']))}"
            # )

    workdir = os.path.join(
        workdir, "evaluations", f"ctx_traj_size_{config.data.known_traj_range}"
    )
    # Create the directory if it doesn't exist
    os.makedirs(workdir, exist_ok=True)
    # Open the file in write mode

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(
            workdir,
            f"evaluation_metrics_sys_id_{sys_id}_forcst_prob_{config.data.foracsting_problem_prob}.csv",
        ), index=True)

