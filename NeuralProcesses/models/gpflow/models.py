from abc import ABC
from trieste.models.gpflow import GaussianProcessRegression
from gpflow.models import GPR
from gpflow.models.gpr import check_shapes
from trieste.models.gpflow.interface import GPflowPredictor
from trieste.models.gpflow.models import tf, inherit_check_shapes, Dataset, \
    GaussianProcessRegression, TensorType, Optional, Optimizer, OptimizeResult, \
        randomize_hyperparameters, squeeze_hyperparameters, read_values, multiple_assign, VariationalGaussianProcess
import jax
from jax import random
from jax import numpy as np
from functools import partial
from jax.typing import ArrayLike
import tensorflow_probability as tfp
from trieste.models.interfaces import ReparametrizationSampler
from NeuralProcesses.data.datasets import NP_Dataset
from flax.training.train_state import TrainState 
from .sampler import MultiOutputBatchReparametrizationSampler
from NeuralProcesses.data.utils import safer_cholesky
from gpflow.utilities import add_likelihood_noise_cov
from gpflow.logdensities import multivariate_normal


class FlaxModel(ABC):
    """
    JAX Flax trained models.
    """
    def predict():
        """
        Predictions of FlaxModel
        """
        pass


class FlaxDummyGPFlowModelWrapper(GaussianProcessRegression):
    """
    Wrapper of Flax model to be used with GPFlow and trieste interfaces for optimization

    The main important thing is we need to keep the functionality of 
    model.update
    model.predict
    
    this needs the GPFlowFlaxModelWrapper to keep it own dataset
    this does not need to have single output
    """
    def __init__(
        self,
        flax_model_predict_func: callable,
        state_dim: int,
        t0: float, 
        t1: float,
        initial_cond_mapper: Optional[callable] = lambda x: x,
        initial_cond_decision_dim: Optional[int] = None,
        optimizer: Optimizer | None = None,
        trajectory_aware: bool = False,
        **kwargs
    ):
        """
        :param flax_model_predict_func: The flax model
        :param state_dim: The state dimension
        :params initial_cond_mapper: sometimes the ODE initial condition can be defined with a lower dimension than the 
            real state dimension, for instance for the SIR model the state num can be defined through 1 value (proportion of infected people)
            hence we introduce this mapper incase the initial condition is not the same as the state dimension
        :params initial_cond_decision_dim: The dimension of the initial condition, if None, it will be the same as the state_dim
        :param optimizer: placeholder, this will not be used for flax models
        :param trajectory_aware: Whether the model is aware of the data consists of trajectories, if not, it will treat each query points totally independent, 
            which is the default option, otherwise, the model will be aware of the trajectory, this is expected to be more computationally efficient, note 
            that this is used in conjuction with the acquisition function, which will provide different query points shape for different trajectory_aware
        """
        self._t0 = t0
        self._t1 = t1
        self._state_dim = state_dim
        self._context_data = None
        self._trajectory_aware = trajectory_aware
        self._model_pred_func = flax_model_predict_func
        self._model_conditional_predict_func_independent = None
        self._initial_cond_mapper = initial_cond_mapper
        if initial_cond_decision_dim is None:
            self.initial_cond_decision_dim = state_dim
        else:
            self.initial_cond_decision_dim = initial_cond_decision_dim
        

    def update_model_conditional_predict_func(self, dataset: NP_Dataset) -> None:
        """
        Given the dataset, return a predictor function that can be used to predict the output
        the predictor function have two forms depending on the kwargs trajectory_aware

        The prediction logic is as follows: stack the new target trajectory at the top of the context data (with only initial condition as the context data), 
        and then use the model to predict the output

        :params dataset: The dataset to be used to update the model
        """
        self._context_data = dataset
        self._context_data.formalize_training_data_sanodep() # this is a must need as otherwise .times will not be initialized

        # in order to ensure that the following function gets updated every time the the update function is called
        # we make them as return of helper functions
        def create_model_predict_with_conditioning_independent():
            @partial(jax.jit, static_argnames=("sample_size"))
            def model_predict_with_conditioning_independent(inputs, rng, sample_size=32):
                """
                this function will treat every inputs independently and use the model to predict the output 

                :params inputs [batch_size, ...]
                :params rng
                :params sample_size
                """
                def seriealize_model_predict_func(initial_cond, target_times):
                    """
                    :params initial_cond [state_dim + 1] where 1 stands for the time
                    :params target_times
                    return [smp, state_dim], [smp, state_dim]
                    """
                    aug_context_time, aug_context_state, aug_context_mask = self._context_data.formalize_training_data_with_pred_init_cond(np.atleast_2d(initial_cond), np.atleast_2d(self._t0))
                    pred_mean_smp, pred_std_smp = self._model_pred_func(aug_context_time, aug_context_state, aug_context_mask, np.atleast_2d(target_times), np.atleast_2d(initial_cond), np.ones_like(target_times, dtype=np.bool_), self._t0, self._t1, rng, sample_size)  
                    return np.squeeze(pred_mean_smp, axis=-2), np.squeeze(pred_std_smp, axis=-2)

                # we use a simple hard code
                # init_cond, times = inputs[..., 0], inputs[..., 1] # [batch_size, state_dim], [batch_size, 1]
                # init_cond = np.stack([2 * init_cond, init_cond], axis=-1)
                # init_cond, times = inputs[..., :self.state_dim], inputs[..., self.state_dim:] # [batch_size, state_dim], [batch_size, timesteps]
                decision_initial_cond, times = inputs[..., :self.initial_cond_decision_dim], inputs[..., self.initial_cond_decision_dim:] # [batch_size, state_dim], [batch_size, timesteps]
                init_cond = decision_initial_cond # self._initial_cond_mapper(decision_initial_cond)
                # debug: 
                # debug_output = seriealize_model_predict_func(init_cond[0], times[0]) # [batch_size, sample_size, state_dim]
                pred_means, pred_stds = jax.vmap(seriealize_model_predict_func, in_axes=(0, 0))(init_cond, times) # [batch_size, sample_size, state_dim]
                return pred_means, pred_stds
            return model_predict_with_conditioning_independent


        def create_model_predict_aware_of_trajectory():
            @partial(jax.jit, static_argnames=("sample_size"))
            def model_predict_aware_of_trajectory(inputs, rng, sample_size=32):
                """
                Model prediction aware of the trajectory
                """
                """
                this function will treat every inputs independently and use the model to predict the output 

                :params context_dataset
                :params inputs
                :params rng
                :params sample_size
                """
                def seriealize_model_predict_func(initial_cond, target_times):
                    """
                    :params initial_cond [state_dim + 1] where 1 stands for the time
                    :params target_times
                    return [smp, batch_size, state_dim], [smp, batch_size, state_dim]
                    """
                    # augment the context data with the initial condition
                    # data class has its own way to handle initial transformation
                    aug_context_time, aug_context_state, aug_context_mask = self._context_data.formalize_training_data_with_pred_init_cond(np.atleast_2d(initial_cond), np.atleast_2d(self._t0))
                    # predict the output (which does not necessarily be the initial condition)
                    # we need to add initial condition transformation here, somewhat ugly
                    pred_mean_smp, pred_std_smp = self._model_pred_func(aug_context_time, aug_context_state, aug_context_mask, np.atleast_2d(target_times), np.atleast_2d(self._initial_cond_mapper(initial_cond)), np.ones_like(target_times, dtype=np.bool_), self._t0, self._t1, rng, sample_size)  
                    return pred_mean_smp, pred_std_smp

                # init_cond, times = inputs[..., :self._state_dim], inputs[..., self._state_dim:] # [batch_size, state_dim], [batch_size, timesteps]
                decision_initial_cond, times = inputs[..., :self.initial_cond_decision_dim], inputs[..., self.initial_cond_decision_dim:] # [batch_size, state_dim], [batch_size, timesteps]
                init_cond = decision_initial_cond
                # pred_means, pred_stds = seriealize_model_predict_func(init_cond[0], times[0]) # debug usage
                pred_means, pred_stds = jax.vmap(seriealize_model_predict_func, in_axes=(0, 0))(init_cond, times) # [batch_size, sample_size, timesteps, state_dim]
                return pred_means, pred_stds
            return model_predict_aware_of_trajectory

        if self._trajectory_aware:
            self._model_conditional_predict_func_independent = create_model_predict_aware_of_trajectory()
        else:
            self._model_conditional_predict_func_independent = create_model_predict_with_conditioning_independent()

    @property
    def model(self) -> FlaxModel:
        return self._model

    def update(self, dataset: NP_Dataset, rng: Optional[random.PRNGKey] = None) -> None:
        """
        Update the flax model datset as well as the model predict
        function
        """
        self.update_model_conditional_predict_func(dataset)


    def optimize(self, dataset: Dataset) -> None:
        """dummy optimize method"""
        return None

    def get_internal_data(self) -> Dataset:
        """
        Return the model's training data.

        :return: The model's training data.
        """
        raise NotImplementedError

    def predict(self, 
                query_points: ArrayLike, 
                sample_size: int = 32, 
                rng: Optional[random.PRNGKey] = random.PRNGKey(0)) -> tuple[ArrayLike, ArrayLike]:
        """
        If trajectory aware, the output will be 
            [batch_size, sample_size, timesteps, state_dim]
        otherwise
            [batch_size, sample_size, state_dim]
        """
        pred_means, _  = self._model_conditional_predict_func_independent(query_points, rng=rng, sample_size=sample_size)
        if self._trajectory_aware: # [batch_size, sample_size, timesteps, state_dim]
            smp_means, smp_vars = np.mean(pred_means, axis=-3), np.std(pred_means, axis=-3) ** 2
        else: # [batch_size, sample_size, state_dim]
            smp_means, smp_vars = np.mean(pred_means, axis=-2), np.std(pred_means, axis=-2) ** 2 
        return smp_means, smp_vars
    
    def sample(self, query_points: ArrayLike, sample_size: int, rng: random.PRNGKey) -> ArrayLike:
        pred_means, _  = self._model_conditional_predict_func_independent(query_points, sample_size=sample_size, rng=rng)
        return pred_means


    def conditional_predict_f_sample(
        self, query_points: TensorType, additional_data: Dataset, num_samples: int
    ) -> TensorType:
        """
        Generates samples of the GP at query_points conditioned on both the model
        and some additional data.

        :param query_points: Set of query points with shape [M, D]
        :param additional_data: Dataset with query_points with shape [..., N, D] and observations
                 with shape [..., N, L]
        :param num_samples: number of samples
        :return: samples of f at query points, with shape [..., num_samples, M, L]
        """
        raise NotImplementedError
    
    def reparam_sampler(self, num_samples: int) -> ReparametrizationSampler[GPflowPredictor]:
        raise NotImplementedError


class MultiFidelityFlaxGPFlowModelWrapper(GaussianProcessRegression):
    """
    This model will use flax model to sample data as low fidelity data, 
    and use linear model of coregionalization to model the high fidelity data
    """


class SampableMultiOutputVariationalGaussianProcess(VariationalGaussianProcess):
    def reparam_sampler(self, num_samples: int) -> ReparametrizationSampler[GPflowPredictor]:
        return MultiOutputBatchReparametrizationSampler(num_samples, self)


class EagerModeGaussianProcessRegression(GaussianProcessRegression):
    """
    Eager mode Gaussian Process Regression
    this model is used to make sure to be trainable together with flax models
    """

    @inherit_check_shapes
    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        # the self._posterior.predict_f is not correct 
        # and hence, we have to make use of self.model to work with deep meta mean
        mean, cov = self.model.predict_f(query_points)
        # mean, cov = (self._posterior or self.model).predict_f(query_points)
        # posterior predict can return negative variance values [cf GPFlow issue #1813]
        if self._posterior is not None:
            cov = tf.clip_by_value(cov, 1e-12, cov.dtype.max)
        return mean, cov
    
    def optimize(self, dataset: Dataset) -> OptimizeResult:
        """
        Optimize the model with the specified `dataset`.

        For :class:`GaussianProcessRegression`, we (optionally) try multiple randomly sampled
        kernel parameter configurations as well as the configuration specified when initializing
        the kernel. The best configuration is used as the starting point for model optimization.

        For trainable parameters constrained to lie in a finite interval (through a sigmoid
        bijector), we begin model optimization from the best of a random sample from these
        parameters' acceptable domains.

        For trainable parameters without constraints but with priors, we begin model optimization
        from the best of a random sample from these parameters' priors.

        For trainable parameters with neither priors nor constraints, we begin optimization from
        their initial values.

        :param dataset: The data with which to optimize the `model`.
        """

        num_trainable_params_with_priors_or_constraints = tf.reduce_sum(
            [
                tf.size(param)
                for param in self.model.trainable_parameters
                if param.prior is not None or isinstance(param.bijector, tfp.bijectors.Sigmoid)
            ]
        )

        # TODO: Make the following compatible with flax
        if (
            min(num_trainable_params_with_priors_or_constraints, self._num_kernel_samples) >= 1
        ):  # Find a promising kernel initialization
            self.find_best_model_initialization(
                self._num_kernel_samples * num_trainable_params_with_priors_or_constraints
            )
        result = self.optimizer.optimize(self.model, dataset)
        self.update_posterior_cache()
        return result
    
    def find_best_model_initialization(self, num_kernel_samples: int) -> None:
        """
        Test `num_kernel_samples` models with sampled kernel parameters. The model's kernel
        parameters are then set to the sample achieving maximal likelihood.

        :param num_kernel_samples: Number of randomly sampled kernels to evaluate.
        """
        # the only change we have in this method is to not us tf.function, which will compile tensors 
        # @tf.function
        def evaluate_loss_of_model_parameters() -> tf.Tensor:
            randomize_hyperparameters(self.model)
            return self.model.training_loss()

        squeeze_hyperparameters(self.model)
        current_best_parameters = read_values(self.model)
        min_loss = self.model.training_loss()

        for _ in tf.range(num_kernel_samples):
            try:
                train_loss = evaluate_loss_of_model_parameters()
            except tf.errors.InvalidArgumentError:  # allow badly specified kernel params
                train_loss = 1e100

            if train_loss < min_loss:  # only keep best kernel params
                min_loss = train_loss
                current_best_parameters = read_values(self.model)

        multiple_assign(self.model, current_best_parameters)


class SaferGaussianProcessRegression(GaussianProcessRegression):
    """
    We use safer cholesky to make robust model performance here
    """

    def optimize(self, dataset: Dataset) -> OptimizeResult:
        """
        Optimize the model with the specified `dataset`.

        Different from the origina one, if optimization failed, it will simply use the best from random start
        as a "grid search" method to find the best hyperparameters
        """

        num_trainable_params_with_priors_or_constraints = tf.reduce_sum(
            [
                tf.size(param)
                for param in self.model.trainable_parameters
                if param.prior is not None or isinstance(param.bijector, tfp.bijectors.Sigmoid)
            ]
        )

        # if (
        #     min(num_trainable_params_with_priors_or_constraints, self._num_kernel_samples) >= 1
        # ):  # Find a promising kernel initialization
        #     self.find_best_model_initialization(
        #         self._num_kernel_samples * num_trainable_params_with_priors_or_constraints
        #     )
        try:
            result = self.optimizer.optimize(self.model, dataset)
        except:
            print("Failed to optimize the model, we will use the randomly sampled hyperparams")
        if not result.success:
            print("Failed to optimize the model, we will use the randomly sampled hyperparams")
            self.find_best_model_initialization(
                self._num_kernel_samples * num_trainable_params_with_priors_or_constraints
            )
        self.update_posterior_cache()
        return result


class Safer_GPR(GPR):
    @check_shapes(
        "return: []",
    )
    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        X, Y = self.data
        K = self.kernel(X)
        # replace this cholesky with safer cholesky
        ks = add_likelihood_noise_cov(K, self.likelihood, X)
        L = safer_cholesky(ks)
        # L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y, m, L)
        return tf.reduce_sum(log_prob)