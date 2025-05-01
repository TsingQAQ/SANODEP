import gpflow
import tensorflow as tf
from trieste.data import Dataset as trieste_Dataset
from tensorflow_probability import distributions as tfd
from NeuralProcesses.models.gpflow.models import SaferGaussianProcessRegression, Safer_GPR
from trieste.models.interfaces import TrainablePredictJointReparamModelStack


def build_model(data: trieste_Dataset):
    variance = 1.0
    kernel = gpflow.kernels.Matern32(
        variance=variance, lengthscales=[1.0] * data.query_points.shape[-1]
    )
    
    prior_scale = tf.cast(2.0, dtype=tf.float64)
    # copied from https://github.com/secondmind-labs/trieste/blob/f67c85c11cf5e411fb796cfe0311e08b1bd8bb10/docs/notebooks/expected_improvement.pct.py#L73
    kernel.variance.prior = tfd.LogNormal(
        tf.cast(-2.0, dtype=tf.float64), prior_scale
    )
    kernel.lengthscales.prior = tfd.LogNormal(
        tf.math.log(kernel.lengthscales), prior_scale
    )
    gpr = Safer_GPR(data.astuple(), kernel, noise_variance=1e-5)
    # specify prior for model to make use of random sampling of hyperparameters as otherwise 
    # the model will break down when Cholesky is not working
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