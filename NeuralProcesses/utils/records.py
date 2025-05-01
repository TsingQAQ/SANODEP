from jax import numpy as np
from ml_collections import ConfigDict


class metrics:
    def __init__(self, dictinaries: dict):
        self.dict = dictinaries

    def append(self, elements):
        self.dict = {
            key: np.concatenate([self.dict[key], np.atleast_1d(val)])
            for key, val in elements.items()
        }

    @property
    def average(self):
        return {key: np.mean(val) for key, val in self.dict.items()}

def initialize_metrics_record(config: ConfigDict, eval: bool = False)->metrics:
    """
    initialize the metrics record for training or evaluation, also used in tqdm progress bar
    :param eval: bool, whether the record is for evaluation, if True, not store the training loss and its related metrics (loss_dict)
    """
    def _init_loss_dict() -> dict:
        loss_dict = {
            "tr_loss": np.zeros(shape=(0,)),
            "nll": np.zeros(shape=(0,)),
            # "target_loss": np.zeros(shape=(0,)),
            "kl_ctrl": np.zeros(shape=(0,)),
            "kl_dnmc": np.zeros(shape=(0,)),
            "kl_z0": np.zeros(shape=(0,)),
        }
        return loss_dict
    
    def _init_metric_dict(metrics_dict: dict) -> dict:
        metric_dict = {
            key: np.zeros(shape=(0,)) for key in metrics_dict.keys()
        }
        return metric_dict
    
    if eval is False:
        # NOTE `loss_dict` is hard coded atm
        if "aux_eval_metric" in config.data:
            aux_dict = _init_metric_dict(config.data.aux_eval_metric)
        else:
            aux_dict = {}
        loss_dict = _init_loss_dict()

        return metrics(loss_dict | aux_dict | {"max_ode_steps": np.zeros(shape=(0,))})
    else:
        metric_dict = _init_metric_dict(config.data.eval_metrics)
        return metrics(metric_dict)
    

def update_metrics(new_metrics: dict, metric_record: metrics) -> metrics:
    """
    format the metrics for tqdm progress bar
    """
    max_ode_steps = np.max(new_metrics.pop("ode_steps"))
    step_metric_collections = (new_metrics | {"max_ode_steps": max_ode_steps})
    
    metric_record.append(step_metric_collections)
    return metric_record