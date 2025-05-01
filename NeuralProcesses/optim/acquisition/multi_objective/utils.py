from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.result import Result as PyMOOResult
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from typing import Callable
from jax import numpy as np
from jax import random
from tqdm import tqdm


from jax.typing import ArrayLike


class MOOResult:
    """
    Wrapper for pymoo result, the main difference the constraint, if have, is >0 is feasible
    """

    def __init__(self, result: PyMOOResult):
        self._res = result

    @property
    def inputs(self):
        return self._res.X

    @property
    def fronts(self):
        """
        return the (feasible) Pareto Front from Pymoo, if no constraint has been satisfied
        """
        return self._res.F


def moo_nsga2_pymoo(
        f: Callable[[], ArrayLike],
        input_dim: int,
        obj_num: int,
        bounds: tuple,
        popsize: int,
        minimum_euc_dist_between_pf_set: float,
        num_generation: int = 1000,
        verbose: bool = False
) -> MOOResult:
    """
    Multi-Objective Optimizer using NSGA2 algorithm by pymoo

    :param f
    :param obj_num
    :param input_dim
    :param bounds: [[lb_0, lb_1, ..., lb_D], [ub_0, ub_1, ..., ub_D]]
    :param popsize: population size for NSGA2
    :param num_generation: number of generations used for NSGA2
    :return 
    """

    def func(x):
        "wrapper objective function for Pymoo written in numpy"
        return f(x)

    class MyProblem(Problem):
        def __init__(self, n_var: int, n_obj: int):
            """
            :param n_var input variables
            :param n_obj number of objective functions
            :param n_constr number of constraint numbers
            """
            super().__init__(
                n_var=n_var,
                n_obj=n_obj,
                xl=bounds[0].numpy(),
                xu=bounds[1].numpy(),
            )

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = func(x)

    problem = MyProblem(n_var=input_dim, n_obj=obj_num)

    algorithm = NSGA2(  # we keep the hyperparameter for NSGA2 fixed here
        pop_size=popsize,
        eliminate_duplicates=DefaultDuplicateElimination(minimum_euc_dist_between_pf_set)
    )

    res = minimize(problem, algorithm, ("n_gen", num_generation), save_history=False, verbose=verbose)

    return MOOResult(res)


# def sample_pareto_fronts_from_meta_learned_models(
#         flax_model, 
#         search_space, 
#         obj_func: Callable,
#         pf_sample_num: int,
#         pf_size: int, 
#         start_rng: random.PRNGKey, 
#         time_scaling: float = 1.0, 
#         num_generation: int = 500) -> tuple[list]:
#     """
#     time_scaling: scaling factor for time dimension (last dimension of the input), 
#     it is important to know that it is assumed that the input space of time starts 
#     from 0, as otherwise the time scaling will not work right
#     """
#     
#     _, rng = random.split(start_rng, 2)
#     pf_samples = []
#     pf_set_samples = []
#     pf_raw_outputs = []
#     for _ in tqdm(range(pf_sample_num)):
#         rng, model_sample_rng = random.split(rng, 2)
#         def multi_obj_fn(x):
#             # within the obj_func, there will be a scaling happens so we still use the scaled time here
#             obj1 = - np.squeeze(obj_func(flax_model.sample(x, sample_size=1, rng=model_sample_rng)), axis=-2) # squeeze out the sample dimensioanlity
#             obj2 = x[..., -1:] * time_scaling # original time
#             return np.concatenate([obj1, obj2], axis=-1)
#         res = moo_nsga2_pymoo(multi_obj_fn, 
#                               input_dim=search_space.lower.shape[0], 
#                               obj_num=2, 
#                               bounds=(search_space.lower, search_space.upper), 
#                               popsize=pf_size, 
#                               minimum_euc_dist_between_pf_set=1e-3, 
#                               num_generation=num_generation)
#         pf_samples.append(res.fronts)
#         pf_set_samples.append(res.inputs)
#         pf_raw_outputs.append(np.squeeze(flax_model.sample(res.inputs, sample_size=1, rng=model_sample_rng), axis=-2))
#     return pf_set_samples, pf_samples, pf_raw_outputs


def sample_pareto_front_from_observer(search_space, 
                                      obj_func: Callable, 
                                      pf_size: int, 
                                      time_scaling: float = 1.0, 
                                      num_generation: int = 1000, 
                                      initial_cond_mapper: Callable = lambda x: x):
    """
    time_scaling: scaling factor for time dimension (last dimension of the input), it is important to know
    that it is assumed that the input space of time starts from 0, as otherwise the time scaling will not work right
    """
    def multi_obj_fn(x):
        # within the obj_func, there will be a scaling happens so we still use the scaled time here
        obj1 = - obj_func(initial_cond_mapper(x[..., :-1]), x[..., -1:]) # squeeze out the sample dimensioanlity
        obj2 = x[..., -1:] * time_scaling # original time
        return np.concatenate([obj1, obj2], axis=-1)
    res = moo_nsga2_pymoo(multi_obj_fn, 
                          input_dim=search_space.lower.shape[0], 
                          obj_num=2, 
                          bounds=(search_space.lower, search_space.upper), 
                          popsize=pf_size, 
                          minimum_euc_dist_between_pf_set=1e-3, 
                          num_generation=num_generation, 
                          verbose = True)
    return res.fronts, res.inputs