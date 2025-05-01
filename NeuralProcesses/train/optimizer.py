import optax

def create_optimizer(config):
    """
    Returns a optax optimizer object based on `config`."""
    # use grad clipping
    return optax.chain(   
        optax.zero_nans(),
        optax.clip_by_global_norm(2.0),
        optax.rmsprop(learning_rate=config.training.optimizer.args.learning_rate))