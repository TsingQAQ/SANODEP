"""Training, evaluation and optimization"""
from NeuralProcesses import train_eval
from NeuralProcesses import plotting
from NeuralProcesses import optimize
from jax import random
# absl can be regarded as an alternative of argparse
# refer; https://999999999.hatenablog.com/entry/argument_parse_with_abseil
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import tensorflow as tf
import logging
import os

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)  # Defines flag for `ConfigDict` files compatible with absl flags

flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum(
    "mode", None, ["train", "eval", "eval_gp", "opt", "plot"], "Running mode: train or eval"
) # enum: a value that can only be either "train" or "evel" or "eval_gp" or "opt" or "plot"
flags.DEFINE_integer(
    "train_seed",
    None,
    "The seed for model initialization",
)  # the seed for model initialization, will not be used for data generation
flags.DEFINE_float(
    "forcast_prob",
    None,
    "The probability of doing forcast task in the training data generation",
)  
flags.DEFINE_enum(
    "model", None, ["meta_learn", "gp"], "Which model to use"
) 
flags.DEFINE_string(
    "eval_folder", "eval", "The folder name for storing evaluation results"
)
flags.DEFINE_enum("opt_type", None, ["so", "mo"], "optimization types")
flags.DEFINE_bool("across_traj_plot", False, "Whether to plot the optimization process")
flags.DEFINE_bool("within_traj_plot", False, "Whether to plot the optimization process")
flags.DEFINE_integer(
    "opt_seed",
    0,
    "A unique integer for the ooptimization experiment, will be used for design of experiments and optimization result filename",
)
flags.DEFINE_integer(
    "process_idx", 
    0,
    "The index of the process for the gp evaluation experiment, will be used for result filename",
)
flags.DEFINE_float("eval_forcast_prob", None, "The probability of doing forcast task in the evaluation data generation")
flags.DEFINE_integer("train_epoch", None, "The number of epochs for training")
flags.DEFINE_bool("loss_comparison", False, "Whether to compare the loss of SANODEP with different loss formulation")
flags.DEFINE_integer("eval_model_smp_size", None, "The number of samples for evaluating the model, mainly used for Neural Processes models")
flags.DEFINE_integer("eval_ctx_traj_size", None, "The number of context trajectories known for evaluating the model, mainly used for Neural Processes models")
flags.DEFINE_integer("eval_step_num", None, "The number of steps for evaluating the model, mainly used for Neural Processes models")
# optional and only used for loss comparison of SANODEP
flags.DEFINE_enum(
    "loss", "NeuralODEProcessMFVILossCondz0ConddsysLoss", 
                  ["NeuralODEProcessMFVILossCondz0ConddsysLoss", 
                   "NeuralODEProcessMFVILossCondz0UconddsysLoss", 
                   "NeuralODEProcessMFVILossUncondz0ConddsysLoss", 
                   "NeuralODEProcessMFVILossUncondz0UnconddsysLoss"], "SANODEP Loss formulation"
)  
from NeuralProcesses.utils.dir_mapping import helper_loss_dir_name_mapping, helper_model_dir_name_mapping


# https://abseil.io/docs/python/quickstart
flags.mark_flags_as_required(
    ["workdir", "config", "mode"]
)  # all the three are required


def main(argv):
    # translate the workdir to the absolute path
    FLAGS.workdir = os.path.abspath(FLAGS.workdir)
    
    if FLAGS.mode == "train":
        # Create the working directory
        tf.io.gfile.makedirs(FLAGS.workdir)
        # Set logger so that it outputs to both console and file
        # Make logging work for both disk and Google Cloud Storage
        gfile_stream = tf.io.gfile.GFile(os.path.join(FLAGS.workdir, "stdout.txt"), "w")
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel("INFO")
        # Run the training pipeline
        # specify the seed here
        if FLAGS.config.model.name == 'SANODEP' and FLAGS.loss_comparison:
            from ml_collections import ConfigDict
            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["loss_method"] = FLAGS.loss
            FLAGS.config = ConfigDict(__mutable_config)
            FLAGS.workdir = os.path.join(FLAGS.workdir, helper_loss_dir_name_mapping[FLAGS.loss])
        if FLAGS.forcast_prob is not None:
            from ml_collections import ConfigDict

            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["data"]["foracsting_problem_prob"] = FLAGS.forcast_prob
            FLAGS.config = ConfigDict(__mutable_config)
            FLAGS.workdir = os.path.join(FLAGS.workdir, f"forcast_prob{FLAGS.forcast_prob}")
        if FLAGS.train_seed is not None:
            from ml_collections import ConfigDict

            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["seed"] = FLAGS.train_seed
            FLAGS.config = ConfigDict(__mutable_config)
            FLAGS.workdir = os.path.join(FLAGS.workdir, f"seed_{FLAGS.train_seed}")
        if FLAGS.train_epoch is not None:
            from ml_collections import ConfigDict

            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["training"]["num_epochs"] = FLAGS.train_epoch
            FLAGS.config = ConfigDict(__mutable_config)
        train_eval.train_meta_learn_model(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "eval":
        # add model name to the workdir
        # FLAGS.workdir = os.path.join(FLAGS.workdir, helper_model_dir_name_mapping[FLAGS.config.model.name])

        if FLAGS.config.model.name == 'SANODEP' and FLAGS.loss_comparison:
            from ml_collections import ConfigDict
            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["loss_method"] = FLAGS.loss
            FLAGS.config = ConfigDict(__mutable_config)
            FLAGS.workdir = os.path.join(FLAGS.workdir, helper_loss_dir_name_mapping[FLAGS.loss])

        if FLAGS.forcast_prob is not None:
            from ml_collections import ConfigDict

            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["data"]["foracsting_problem_prob"] = FLAGS.forcast_prob
            FLAGS.config = ConfigDict(__mutable_config)
            FLAGS.workdir = os.path.join(FLAGS.workdir, f"forcast_prob{FLAGS.forcast_prob}")
        # note that allows that the data generated with a different forcast probability 
        # compared with how it has been trained (i.e., the foracsting_problem_prob above),
        # this allows inspecting what the model behaves on pure forcasting / interpolating task even trained
        # in a hybrid approach
        if FLAGS.eval_forcast_prob is not None:
            from ml_collections import ConfigDict

            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["data"]["foracsting_problem_prob"] = FLAGS.eval_forcast_prob
            FLAGS.config = ConfigDict(__mutable_config)
        if FLAGS.train_seed is not None:
            from ml_collections import ConfigDict

            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["seed"] = FLAGS.train_seed
            FLAGS.config = ConfigDict(__mutable_config)
            FLAGS.workdir = os.path.join(FLAGS.workdir, f"seed_{FLAGS.train_seed}")
        if FLAGS.eval_model_smp_size is not None:
            from ml_collections import ConfigDict

            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["model"]["sample_size"] = FLAGS.eval_model_smp_size
            FLAGS.config = ConfigDict(__mutable_config)
        if FLAGS.eval_step_num is not None:
            from ml_collections import ConfigDict

            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["evaluation"]["num_steps"] = FLAGS.eval_step_num
            FLAGS.config = ConfigDict(__mutable_config)
        if FLAGS.eval_ctx_traj_size is not None:
            from ml_collections import ConfigDict

            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["data"]["known_traj_range"] = (FLAGS.eval_ctx_traj_size, FLAGS.eval_ctx_traj_size)
            FLAGS.config = ConfigDict(__mutable_config)
        # Run the evaluation pipeline
        train_eval.eval_meta_learn_model(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "eval_gp":

        if FLAGS.eval_forcast_prob is not None:
            print("eval_forcast_prob", FLAGS.eval_forcast_prob)
            from ml_collections import ConfigDict

            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["data"]["foracsting_problem_prob"] = FLAGS.eval_forcast_prob
            FLAGS.config = ConfigDict(__mutable_config)
        if FLAGS.train_seed is not None:
            from ml_collections import ConfigDict

            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["seed"] = FLAGS.train_seed
            FLAGS.config = ConfigDict(__mutable_config)
            FLAGS.workdir = os.path.join(FLAGS.workdir, f"seed_{FLAGS.train_seed}")
        if FLAGS.eval_step_num is not None:
            from ml_collections import ConfigDict

            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["evaluation"]["num_steps"] = FLAGS.eval_step_num
            FLAGS.config = ConfigDict(__mutable_config)
        if FLAGS.eval_ctx_traj_size is not None:
            from ml_collections import ConfigDict

            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["data"]["known_traj_range"] = (FLAGS.eval_ctx_traj_size, FLAGS.eval_ctx_traj_size)
            FLAGS.config = ConfigDict(__mutable_config)
        # Run the evaluation pipeline
        train_eval.eval_gp_model(FLAGS.config, FLAGS.workdir, sys_id = FLAGS.process_idx)
    elif FLAGS.mode == "plot":
        # add model name to the workdir
        # FLAGS.workdir = os.path.join(FLAGS.workdir, helper_model_dir_name_mapping[FLAGS.config.model.name])
        FLAGS.workdir = os.path.dirname(FLAGS.workdir)
        if FLAGS.config.model.name == 'SANODEP' and FLAGS.loss_comparison:
            from ml_collections import ConfigDict
            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["loss_method"] = FLAGS.loss
            FLAGS.config = ConfigDict(__mutable_config)
            FLAGS.workdir = os.path.join(FLAGS.workdir, helper_loss_dir_name_mapping[FLAGS.loss])

        if FLAGS.forcast_prob is not None:
            from ml_collections import ConfigDict

            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["data"]["foracsting_problem_prob"] = FLAGS.forcast_prob
            FLAGS.config = ConfigDict(__mutable_config)
            FLAGS.workdir = os.path.join(FLAGS.workdir, f"forcast_prob{FLAGS.forcast_prob}")
        # note that allows that the data generated with a different forcast probability 
        # compared with how it has been trained (i.e., the foracsting_problem_prob above),
        # this allows inspecting what the model behaves on pure forcasting / interpolating task even trained
        # in a hybrid approach
        if FLAGS.eval_forcast_prob is not None:
            from ml_collections import ConfigDict

            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["data"]["foracsting_problem_prob"] = FLAGS.eval_forcast_prob
            FLAGS.config = ConfigDict(__mutable_config)
        if FLAGS.train_seed is not None:
            from ml_collections import ConfigDict

            __mutable_config = FLAGS.config.to_dict()
            __mutable_config["seed"] = FLAGS.train_seed
            FLAGS.config = ConfigDict(__mutable_config)
            FLAGS.workdir = os.path.join(FLAGS.workdir, f"seed_{FLAGS.train_seed}")
        # Run the evaluation pipeline
        plotting.plot_meta_learn_model(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "opt":
        if FLAGS.model == "gp":
            optimize.gp_based_optimization(
                FLAGS.config,
                FLAGS.workdir,
                FLAGS.opt_type,
                FLAGS.opt_seed,
                FLAGS.across_traj_plot,
                FLAGS.within_traj_plot,
            )
        elif FLAGS.model == "meta_learn":
            if FLAGS.train_seed is not None:
                from ml_collections import ConfigDict

                __mutable_config = FLAGS.config.to_dict()
                __mutable_config["seed"] = FLAGS.train_seed
            if FLAGS.forcast_prob is not None:
                from ml_collections import ConfigDict
    
                __mutable_config = FLAGS.config.to_dict()
                __mutable_config["data"]["foracsting_problem_prob"] = FLAGS.forcast_prob
                FLAGS.config = ConfigDict(__mutable_config)
                FLAGS.workdir = os.path.join(FLAGS.workdir, f"forcast_prob{FLAGS.forcast_prob}")
                FLAGS.config = ConfigDict(__mutable_config)
                FLAGS.workdir = os.path.join(FLAGS.workdir, f"seed_{FLAGS.train_seed}")
            optimize.meta_bayesian_optimization(
                FLAGS.config,
                FLAGS.workdir,
                FLAGS.opt_type,
                FLAGS.opt_seed,
                FLAGS.across_traj_plot,
                FLAGS.within_traj_plot
            )
        elif FLAGS.model == "adapt_meta_learn":
            train_eval.multi_fidelity_gp_based_optimization(
                FLAGS.config, FLAGS.workdir, FLAGS.opt_type
            )
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    app.run(main)
