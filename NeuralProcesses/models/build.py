from flax.linen import Module
from jax import random
from . import model as model_collections
from ml_collections import ConfigDict
import jax



def init_model(init_rng: random.PRNGKey, config: ConfigDict):
    """ 
    Initialize a `flax.linen.Module` model. 
  
    When the model is an instantioation of DynamicSystemModel, we need to wrap it within a NeuralODEWrapper (and its child class)
    
    return
    :param model: a Frozendict representing model structure hyperparameters
    :param init_model_state:
    :param initial model parameters
    :param rng
    """
    model_name = config.model.name
    # create model instance
    model: Module = getattr(model_collections, model_name)(**config.model.cfg_args)
    # how to unify input then
    params_rng, rng = jax.random.split(init_rng, num=2)
    initial_params = model.init(params_rng, **config.model.init_args) 
    return model, initial_params, rng