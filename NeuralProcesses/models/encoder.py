from flax.linen import Sequential, Dense, relu, Module, sigmoid, softplus
from flax.linen import activation
from jax import numpy as np
from .utils.linalg import create_lower_tri_matrix, calc_elmnt_num_in_ltri_from_dim
from .utils.transformation import transform_diagonal


class Encoder(Module):
    h_dim: int
    o_dim: int
    act_fn: str = 'relu'
    """
    Parameters
    ----------
    h_dim : int
        Dimension of hidden layer.

    o_dim : int
        Dimension of output representation.
    """
    def setup(self):

        self.input_to_hidden = Sequential([Dense(self.h_dim),
                  getattr(activation, self.act_fn),
                  Dense(self.h_dim),
                  getattr(activation, self.act_fn),
                  Dense(self.o_dim)])

    def __call__(self, inputs, mask = None):
        """
        inputs : Shape (batch_size, dim)
        """
        return self.input_to_hidden(inputs)


class MuSigmaEncoder(Module):
    h_dim: int
    o_dim: int
    std_lower_bound: float = 0.1
    act_fn: str = 'relu'
    hidden_layers: int = 1
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    h_dim : int
        Dimension of hidden layer.

    o_dim : int
        Dimension of distribution of output variable.
    std_lower_bound: float
        Minimum predictive standard deviation
    """
    def setup(self):
        hidden_layers = []
        for _ in range(self.hidden_layers):
            hidden_layers.append(Dense(self.h_dim))
            hidden_layers.append(getattr(activation, self.act_fn))
        self.hidden = Sequential(hidden_layers) # Sequential([Dense(self.h_dim), getattr(activation, self.act_fn)])
        self.mu = Dense(self.o_dim)
        self.sigma_layer = Sequential([Dense(self.o_dim), sigmoid])

    def __call__(self, inputs):
        """
        r : [batch_size(optional), r_dim]
        """
        hidden = self.hidden(inputs)
        encoded_mu = self.mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        encoded_sigma = self.std_lower_bound + 0.9 * self.sigma_layer(hidden)
        return encoded_mu, encoded_sigma


class MuSigmaFixedLikelihoodEncoder(Module):
    h_dim: int
    o_dim: int
    std: float
    def setup(self):
        self.hidden = Sequential([Dense(self.h_dim), relu])
        self.mu = Dense(self.o_dim)

    def __call__(self, inputs):
        """
        r : [batch_size(optional), r_dim]
        """
        hidden = self.hidden(inputs)
        encoded_mu = self.mu(hidden)
        return encoded_mu, np.ones_like(encoded_mu) * self.std
    
