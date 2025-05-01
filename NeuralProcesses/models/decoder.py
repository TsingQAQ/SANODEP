from jax import numpy as np
from flax.linen import activation
from flax.linen import Module, Sequential, Dense, relu, softplus
from jax import value_and_grad
from jax import vmap


class FixedLikelihoodDecoder(Module):
    x_dim: int
    z_dim: int
    h_dim: int
    y_dim: int
    std: float = 0.01
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer.

    y_dim : int
        Dimension of y values.

    std: float
         predictive standard deviation
    """
    def setup(self):
        self.xz_to_hidden = Sequential([
                  Dense(self.h_dim),
                  relu,
                  Dense(self.h_dim),
                  relu,
                  Dense(self.h_dim),
                  relu])
        # I thought each z correspond to a single realization!
        # there is a Gaussian noise based
        self.hidden_to_mu = Dense(self.y_dim)

    def __call__(self, x, z):
        """
        x : Shape (num_points, x_dim)

        z : Shape (sample_size, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (sample_size, num_points, y_dim).
        """
        num_points, _ = x.shape
        # batch_size, num_points, _ = x.shape
        num_samples = z.shape[-2]
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        x = np.repeat(np.expand_dims(x, axis=0), num_samples, axis=0)
        z = np.repeat(np.expand_dims(z, axis=1), num_points, axis=1) # expand num_points dimension
        # Input is concatenation of z with every row of x
        input_pairs = np.concatenate([x, z], axis=-1) # [sample_size, num_points, dimensionality]
        hidden = self.xz_to_hidden(input_pairs)
        mu = self.hidden_to_mu(hidden)
        return mu, np.ones_like(mu) * self.std


class HeteroscedasticDecoder(Module):
    x_dim: int
    z_dim: int
    h_dim: int
    y_dim: int
    std_lower_bound: float = 0.1
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer.

    y_dim : int
        Dimension of y values.

    std_lower_bound: float
        Minimum predictive standard deviation
    """
    def setup(self):
        self.xz_to_hidden = Sequential([
                  Dense(self.h_dim),
                  relu,
                  Dense(self.h_dim),
                  relu,
                  Dense(self.h_dim),
                  relu])
        # I thought each z correspond to a single realization!
        # there is a Gaussian noise based
        self.hidden_to_mu = Dense(self.y_dim)
        self.hidden_to_sigma = Dense(self.y_dim) # assume heterschodastic noise

    def __call__(self, x, z):
        """
        x : 
            Shape (num_points, x_dim)

        z : 
            Shape (sample_size, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (sample_size, num_points, y_dim).
        """
        num_points, _ = x.shape
        # batch_size, num_points, _ = x.shape
        num_samples = z.shape[-2]
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        # x = np.repeat(np.expand_dims(x, axis=-2), num_samples, axis=-2) # expand sample dimension
        x = np.repeat(np.expand_dims(x, axis=0), num_samples, axis=0)
        z = np.repeat(np.expand_dims(z, axis=1), num_points, axis=1) # expand num_points dimension
        # z = np.repeat(np.expand_dims(np.repeat(np.expand_dims(z, axis=-2), num_points, axis=-3), axis=0),  batch_size, axis=0) # expand batch dimension
        # Flatten x and z to fit with linear layer
        # x_flat = x.view(batch_size * num_points, self.x_dim)
        # z_flat = z.view(batch_size * num_points, self.z_dim)
        # Input is concatenation of z with every row of x
        input_pairs = np.concatenate([x, z], axis=-1) # [sample_size, num_points, dimensionality]
        hidden = self.xz_to_hidden(input_pairs)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        # Reshape output into expected shape
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = self.std_lower_bound + 0.9 * softplus(pre_sigma)
        return mu, sigma


class EmpiricalMeanVarianceDecoder(Module):
    x_dim: int
    z_dim: int
    h_dim: int
    y_dim: int
    def setup(self):
        self.xz_to_output = Sequential([
                  Dense(self.h_dim),
                  relu,
                  Dense(self.h_dim),
                  relu,
                  Dense(self.h_dim),
                  relu, 
                  Dense(self.y_dim)])

    def __call__(self, x, z):
        """
        x : Shape (num_points, x_dim)

        z : Shape (sample_size, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (sample_size, num_points, y_dim).
        """
        num_points, _ = x.shape
        # batch_size, num_points, _ = x.shape
        num_samples = z.shape[-2]
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        x = np.repeat(np.expand_dims(x, axis=0), num_samples, axis=0)
        z = np.repeat(np.expand_dims(z, axis=1), num_points, axis=1) # expand num_points dimension
        # Input is concatenation of z with every row of x
        input_pairs = np.concatenate([x, z], axis=-1) # [sample_size, num_points, dimensionality]
        output = self.xz_to_output(input_pairs)

        return np.mean(output, axis=0, keepdims=True), np.std(output, axis=0, keepdims=True), output


class LinearizeDecoder(Module):
    """
    The main idea about this decoder is to use Laplace approximation to extract the uncertainty 
    """
    x_dim: int
    z_dim: int
    h_dim: int
    y_dim: int
    def setup(self):
        self.xz_to_output = Sequential([
                  Dense(self.h_dim),
                  relu,
                  Dense(self.h_dim),
                  relu,
                  Dense(self.h_dim),
                  relu, 
                  Dense(self.y_dim)])

    def __call__(self, x, z, z_mean, z_cov):
        """
        x : Shape (num_points, x_dim)

        z : Shape (sample_size, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (sample_size, num_points, y_dim).
        """
        num_points, _ = x.shape
        # batch_size, num_points, _ = x.shape
        num_samples = z.shape[-2]
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        x = np.repeat(np.expand_dims(x, axis=0), num_samples, axis=0)
        z = np.repeat(np.expand_dims(z, axis=1), num_points, axis=1) # expand num_points dimension
        # Input is concatenation of z with every row of x
        # input_pairs = np.concatenate([x, z], axis=-1) # [sample_size, num_points, dimensionality]
        # output = self.xz_to_output(input_pairs)
        # mean = np.squeeze(output, axis=0)
        def decoder(x, z): # TODO: Only support one output atm
            input_pairs = np.concatenate([x, z], axis=-1) # [sample_size, num_points, dimensionality]
            return np.squeeze(self.xz_to_output(input_pairs))
        # Assuming decoder is your function and it returns a scalar for each instance in the batch
        # Compute the value and gradient for each instance in the batch
        value_and_grad_vmap = vmap(vmap(value_and_grad(decoder, argnums=1), in_axes=(0, None)), in_axes=(0, None))

        input_pairs = np.concatenate([x, z], axis=-1) # [sample_size, num_points, dimensionality]
        output = self.xz_to_output(input_pairs)
        # Now compute the value and gradient
        output_mean, grad_z_x = value_and_grad_vmap(x, z_mean)
        # output, grad_z_x = value_and_grad(decoder, argnums=1)(x, z)
        mean = output_mean[..., None] # [1, num_points]
        std = np.sqrt(np.squeeze(np.matmul(np.matmul(np.expand_dims(grad_z_x, axis=-2), z_cov), grad_z_x[..., None]), axis=-2))

        return mean, std, output


class FixedLikelihoodNODEPDecoder(Module):
    h_dim: int
    x_dim: int
    std: float = 0.01
    autonomous_ode: bool = False
    act_fn: str = 'relu'
    """
    Try to implemnt NDP-L
    """
    def setup(self):
        # self.tlz_to_hidden = Sequential([Dense(self.h_dim), relu, Dense(self.h_dim), relu, Dense(self.h_dim), relu])
        # 2024/02/20 changed it to be silu for continuous differentiability
        _act_fn = getattr(activation, self.act_fn)
        self.tlz_to_hidden = Sequential([Dense(self.h_dim), _act_fn, Dense(self.h_dim), _act_fn, Dense(self.h_dim), _act_fn])
        self.hidden_to_mu = Dense(self.x_dim)

    def __call__(self, target_t, z_D, sampled_control):
        """
        target_t: [timesteps]
        z: [sample_size, timesteps, D_dim]
        sampled_control: [sample_size, L_dim]

        return [sample_size, time_steps, x_dim]
        """
        expand_sampled_control = np.repeat(np.expand_dims(sampled_control, axis=-2), target_t.shape[-1], axis=-2)
        if not self.autonomous_ode:
            expand_target_t = np.broadcast_to(np.expand_dims(np.expand_dims(target_t, axis=-1), axis=0), z_D.shape[:-1] + (1,))
            hidden = self.tlz_to_hidden(np.concatenate([expand_target_t, z_D, expand_sampled_control], axis=-1))
        else:
            hidden = self.tlz_to_hidden(np.concatenate([z_D, expand_sampled_control], axis=-1))
        hidden = np.concatenate([z_D, hidden], axis=-1) # what is going on here?
        mu = self.hidden_to_mu(hidden)
        return mu, np.ones_like(mu) * self.std


class HeteroscedasticNODEPDecoder(Module):
    h_dim: int
    x_dim: int
    std_lower_bound: float = 0.1
    autonomous_ode: bool = False
    act_fn: str = 'relu'
    """
    Try to implemnt NDP-L
    """
    def setup(self):
        # self.tlz_to_hidden = Sequential([Dense(self.h_dim), relu, Dense(self.h_dim), relu, Dense(self.h_dim), relu])
        # 2024/02/20 changed it to be silu for continuous differentiability
        _act_fn = getattr(activation, self.act_fn)
        self.tlz_to_hidden = Sequential([Dense(self.h_dim), _act_fn, Dense(self.h_dim), _act_fn, Dense(self.h_dim), _act_fn])
        self.hidden_to_mu = Dense(self.x_dim)
        self.hidden_to_sigma = Dense(self.x_dim)

    def __call__(self, target_t, z_D, sampled_control):
        """
        target_t: [..., timesteps]
        z: [..., sample_size, timesteps, D_dim]
        sampled_control: [..., sample_size, L_dim]

        return [..., sample_size, time_steps, x_dim]
        """
        
        expand_sampled_control = np.repeat(np.expand_dims(sampled_control, axis=-2), target_t.shape[-1], axis=-2)
        if not self.autonomous_ode:
            expand_target_t = np.broadcast_to(np.expand_dims(np.expand_dims(target_t, axis=-1), axis=0), z_D.shape[:-1] + (1,))
            hidden = self.tlz_to_hidden(np.concatenate([expand_target_t, z_D, expand_sampled_control], axis=-1))
        else:
            hidden = self.tlz_to_hidden(np.concatenate([z_D, expand_sampled_control], axis=-1))
        hidden = np.concatenate([z_D, hidden], axis=-1) # what is going on here?
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)

        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = self.std_lower_bound + 0.9 * softplus(pre_sigma)
        return mu, sigma


class FixedLikelihoodNODEPDecoder_with_global_dynamic(FixedLikelihoodNODEPDecoder):

    def __call__(self, target_t, z_D, sampled_control, global_dynamic):
        """
        target_t: [timesteps]
        z: [traj_size, sample_size, timesteps, D_dim]
        sampled_control: [traj_size, sample_size, L_dim]
        global_dynamic: [traj_size, sample_size, L_dim]
        return [traj_size, sample_size, time_steps, x_dim]
        """
        expand_global_dynamic = np.repeat(np.expand_dims(global_dynamic, -2), axis=-2, repeats=z_D.shape[-2]) # [traj_size, sample_size, timesteps, L_dim]
        expand_sampled_control = np.repeat(np.expand_dims(sampled_control, axis=-2), target_t.shape[-1], axis=-2)
        if not self.autonomous_ode:
            expand_target_t = np.broadcast_to(np.expand_dims(np.expand_dims(target_t, axis=-1), axis=0), z_D.shape[:-1] + (1,))
            hidden = self.tlz_to_hidden(np.concatenate([expand_target_t, z_D, expand_sampled_control, expand_global_dynamic], axis=-1))
        else:
            hidden = self.tlz_to_hidden(np.concatenate([z_D, expand_sampled_control, expand_global_dynamic], axis=-1))
        hidden = np.concatenate([z_D, hidden], axis=-1) # what is going on here?
        mu = self.hidden_to_mu(hidden)
        return mu, np.ones_like(mu) * self.std
    

class HeteroscedasticNODEPDecoder_with_global_dynamic(HeteroscedasticNODEPDecoder):
    def __call__(self, target_t, z_D, sampled_control, global_dynamic):
        """
        target_t: [timesteps]
        z: [sample_size, timesteps, D_dim]
        sampled_control: [sample_size, L_dim]

        return [sample_size, time_steps, x_dim]
        """
        expand_global_dynamic = np.repeat(np.expand_dims(global_dynamic, -2), axis=-2, repeats=z_D.shape[-2])
        expand_sampled_control = np.repeat(np.expand_dims(sampled_control, axis=-2), target_t.shape[-1], axis=-2)
        if not self.autonomous_ode:
            expand_target_t = np.broadcast_to(np.expand_dims(np.expand_dims(target_t, axis=-1), axis=0), z_D.shape[:-1] + (1,))
            hidden = self.tlz_to_hidden(np.concatenate([expand_target_t, z_D, expand_sampled_control, expand_global_dynamic], axis=-1))
        else:
            hidden = self.tlz_to_hidden(np.concatenate([z_D, expand_sampled_control, expand_global_dynamic], axis=-1))

        hidden = np.concatenate([z_D, hidden], axis=-1) # what is going on here?
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)

        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = self.std_lower_bound + 0.9 * softplus(pre_sigma)
        return mu, sigma


class PI_HeteroscedasticNODEPDecoder(HeteroscedasticNODEPDecoder):
    def __call__(self, target_t, z_D, sampled_params):
        """
        The physical structured informed model's decoder, the only difference is that the mu is not calculated but
         is directly extracted from z_D
        target_t: [..., timesteps]
        z: [..., sample_size, timesteps, D_dim]
        sampled_control: [..., sample_size, L_dim]

        return [..., sample_size, time_steps, x_dim]
        """
        sampled_params = np.repeat(np.expand_dims(sampled_params, axis=-2), target_t.shape[-1], axis=-2)
        if not self.autonomous_ode:
            expand_target_t = np.broadcast_to(np.expand_dims(np.expand_dims(target_t, axis=-1), axis=0), z_D.shape[:-1] + (1,))
            hidden = self.tlz_to_hidden(np.concatenate([expand_target_t, z_D, sampled_params], axis=-1))
        else:
            hidden = self.tlz_to_hidden(np.concatenate([z_D, sampled_params], axis=-1))
        hidden = np.concatenate([z_D, hidden], axis=-1) # what is going on here?
        mu = z_D
        pre_sigma = self.hidden_to_sigma(hidden)

        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = self.std_lower_bound + 0.9 * softplus(pre_sigma)
        return mu, sigma


class PredictVarianceNODEPDecoder(HeteroscedasticNODEPDecoder):
    def __call__(self, target_t, z_D):
        """
        target_t: [..., timesteps]
        z: [..., sample_size, timesteps, D_dim]

        return [..., sample_size, time_steps, x_dim]
        """
        if not self.autonomous_ode:
            expand_target_t = np.broadcast_to(np.expand_dims(np.expand_dims(target_t, axis=-1), axis=0), z_D.shape[:-1] + (1,))
            hidden = self.tlz_to_hidden(np.concatenate([expand_target_t, z_D], axis=-1))
        else:
            hidden = self.tlz_to_hidden(np.concatenate([z_D], axis=-1))
        hidden = np.concatenate([z_D, hidden], axis=-1) # what is going on here?
        mu = z_D
        pre_sigma = self.hidden_to_sigma(hidden)

        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = self.std_lower_bound + 0.9 * softplus(pre_sigma)
        return mu, sigma
