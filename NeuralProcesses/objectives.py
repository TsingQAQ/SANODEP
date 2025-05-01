"""
Construct the objective function to be maximized

For now most of the objective functions is formulated from state variables together with time
One thing to remark here is, for problems like Lotka Voterra system, 
the model, follows the NODEP's convention, is built upon a scaled time range by parameter time_scaling_coefficient
which is used to generate the meta dataset

To calculate the objectives when time_scaling_coefficient is not 1, we need to multiply the time by time_scaling_coefficient 
to get the correct time
"""
from diffrax import diffeqsolve, ODETerm, PIDController, Dopri5, SaveAt
from jax import numpy as np
from jax import random
from jax import jit
from einops import rearrange
import tensorflow as tf
from jax import vmap
import jax



def obj_func_1d(states):
    """
    Note that this must be of tensorflow form to be used in the acquisition function
    """
    return states


def lotka_voterra_obj_func_2d(states):
    """
    Note that this must be of tensorflow form to be used in the acquisition function
    """
    return states[..., :1]

@jit
def lotka_voterra_1d_observer(init_cond, times, t0, t1, problem_rng: random.PRNGKey, time_scaling: float = 1.0):
    def ode():        
        alpha =   0.5 
        beta = 1.2 
        delta = 1.0  
        gamma =  1.5   
        def dynamics(t, _x, args):
            u, v = _x[..., 0], _x[..., 1]
            return np.stack([alpha * u - beta * u * v, delta * u * v - gamma * v], axis=-1)
        # solve ode
        return dynamics
    E = np.atleast_1d(init_cond)
    # x0 = np.concatenate([2 * E, E], axis=-1) # [dynamic_smp, ibatch_size, 2]
    x0 = np.concatenate([E, E], axis=-1) # [dynamic_smp, ibatch_size, 2]
    # x0 = init_cond
    # times = times * 10 # use the same as in Loteka-Volterra
    times = times * time_scaling
    # t1 = t1 * 10 # use the same as in Loteka-Volterra
    t1 = t1 * time_scaling
    state = diffeqsolve(ODETerm(ode()), t0 = t0, t1 = t1, dt0 = None, 
                   stepsize_controller = PIDController(rtol=1e-5, atol=1e-6), y0=x0, 
                   solver=Dopri5(), max_steps=1000, saveat=SaveAt(ts=times)).ys
    return state[..., :1]

@jit
def lotka_voterra_2d_observer(init_cond, times, t0, t1, problem_rng: random.PRNGKey, alpha_range, beta_range, delta_range, gamma_range, 
                              use_fixed_problem: bool = True, time_scaling: float = 1.0):

    if use_fixed_problem:
        alpha = 0.5
        beta = 1.2
        delta = 1.0
        gamma = 1.5
    else:  
        alpha =   random.uniform(problem_rng, (1,), alpha_range[0], alpha_range[1])[0] # 0.5 # 1/3 # 0.9  #
        beta = random.uniform(problem_rng, (1,), beta_range[0], beta_range[1])[0] # 1.2 # 2 # 1.1 #  
        delta = random.uniform(problem_rng, (1,), delta_range[0], delta_range[1])[0] # 1.0   # 0.5 # 1.4 #  
        gamma =  random.uniform(problem_rng, (1,), gamma_range[0], gamma_range[1])[0] # 1.5    # 1.0 # 0.5 #

    def dynamics(t, _x, args):
        u, v = _x[..., 0], _x[..., 1]
        return np.stack([alpha * u - beta * u * v, delta * u * v - gamma * v], axis=-1)

    x0 = init_cond
    # times = times * 10 # use the same as in Loteka-Volterra
    times = times * time_scaling
    # t1 = t1 * 10 # use the same as in Loteka-Volterra
    t1 = t1 * time_scaling
    if len(x0.shape) == 1:
        state = diffeqsolve(ODETerm(dynamics), t0 = t0, t1 = t1, dt0 = None, 
                       stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), y0=x0, 
                       solver=Dopri5(), max_steps=1000, saveat=SaveAt(ts=times)).ys   
        return state
    else:
        batched_diffeqsolve = jax.vmap(
            lambda _x0, _times: diffeqsolve(
                ODETerm(dynamics), 
                t0=t0, 
                t1=t1, 
                dt0=None, 
                stepsize_controller=PIDController(rtol=1e-7, atol=1e-9), 
                y0=_x0, 
                solver=Dopri5(), 
                max_steps=1000, 
                saveat=SaveAt(ts=_times)
            ).ys
        )
        state = batched_diffeqsolve(x0, times)
        return np.squeeze(state, axis=-2)

@jit
def lotka_voterra_3d_observer(init_cond, 
                              times, 
                              t0, 
                              t1, 
                              problem_rng: random.PRNGKey, 
                              alpha_range, 
                              beta_range, 
                              delta_range, 
                              gamma_range, 
                              epsilon_range, 
                              zeta_range, 
                              eta_range,
                              theta_range,
                              use_fixed_problem: bool = True, 
                              time_scaling: float = 1.0):

    if use_fixed_problem:
        alpha = 0.5
        beta = 1.2
        delta = 1.0
        gamma = 1.5
        epsilon = 0.5
        zeta = 1.2
        eta = 1.0
        theta = 1.5
    else:  
        alpha =   random.uniform(problem_rng, (1,), alpha_range[0], alpha_range[1])[0] # 0.5 # 1/3 # 0.9  #
        beta = random.uniform(problem_rng, (1,), beta_range[0], beta_range[1])[0] # 1.2 # 2 # 1.1 #  
        delta = random.uniform(problem_rng, (1,), delta_range[0], delta_range[1])[0] # 1.0   # 0.5 # 1.4 #  
        gamma =  random.uniform(problem_rng, (1,), gamma_range[0], gamma_range[1])[0] # 1.5    # 1.0 # 0.5 #
        epsilon = random.uniform(problem_rng, (1,), epsilon_range[0], epsilon_range[1])[0]
        zeta = random.uniform(problem_rng, (1,), zeta_range[0], zeta_range[1])[0]
        eta = random.uniform(problem_rng, (1,), eta_range[0], eta_range[1])[0]
        theta = random.uniform(problem_rng, (1,), theta_range[0], theta_range[1])[0]

    def dynamics(t, _x, args):
        x_1, x_2, x_3 = _x[..., 0], _x[..., 1], _x[..., 2]
        return np.stack([
            alpha * x_1 - beta * x_1 * x_2 - epsilon * x_1 * x_3,
            delta * x_1 * x_2 - gamma * x_2 - zeta * x_2 * x_3,
            eta * x_1 * x_3 + theta * x_2 * x_3 - gamma * x_3
        ], axis=-1)

    x0 = init_cond
    # times = times * 10 # use the same as in Loteka-Volterra
    times = times * time_scaling
    # t1 = t1 * 10 # use the same as in Loteka-Volterra
    t1 = t1 * time_scaling
    if len(x0.shape) == 1:
        state = diffeqsolve(ODETerm(dynamics), t0 = t0, t1 = t1, dt0 = None, 
                       stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), y0=x0, 
                       solver=Dopri5(), max_steps=1000, saveat=SaveAt(ts=times)).ys   
        return state
    else:
        batched_diffeqsolve = jax.vmap(
            lambda _x0, _times: diffeqsolve(
                ODETerm(dynamics), 
                t0=t0, 
                t1=t1, 
                dt0=None, 
                stepsize_controller=PIDController(rtol=1e-7, atol=1e-9), 
                y0=_x0, 
                solver=Dopri5(), 
                max_steps=1000, 
                saveat=SaveAt(ts=_times)
            ).ys
        )
        state = batched_diffeqsolve(x0, times)
        return np.squeeze(state, axis=-2)

def lotka_voterra_obj_func_3d(states):
    return states[..., :1]


@jit
def sir_unnormalized_observer(init_cond, times, t0, t1, problem_rng: random.PRNGKey, beta_range, gamma_range, use_fixed_problem: bool = True, time_scaling: float = 1.0):
    if use_fixed_problem:
        beta = 1.5
        gamma = 5.0
    else:  
        beta = random.uniform(problem_rng, (1,), beta_range[0], beta_range[1])[0]
        gamma = random.uniform(problem_rng, (1,), gamma_range[0], gamma_range[1])[0]

    times = times * time_scaling
    t1 = t1 * time_scaling
    x0 = init_cond

    def sir_dynamics(t, y, args):
        normed_S, normed_I, normed_R = y[..., 0], y[..., 1], y[..., 2] 
        dSdt = -beta * normed_S * normed_I 
        dIdt = beta * normed_S * normed_I  - gamma * normed_I
        dRdt = gamma * normed_I
        return np.stack([dSdt, dIdt, dRdt], axis=-1)

    if len(x0.shape) == 1:
        state = diffeqsolve(ODETerm(sir_dynamics), t0 = t0, t1 = t1, dt0 = None, 
                       stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), y0=x0, 
                       solver=Dopri5(), max_steps=1000, saveat=SaveAt(ts=times)).ys   
        return state
    else:
        batched_diffeqsolve = jax.vmap(
            lambda _x0, _times: diffeqsolve(
                ODETerm(sir_dynamics), 
                t0=t0, 
                t1=t1, 
                dt0=None, 
                stepsize_controller=PIDController(rtol=1e-7, atol=1e-9), 
                y0=_x0, 
                solver=Dopri5(), 
                max_steps=1000, 
                saveat=SaveAt(ts=_times)
            ).ys
        )
        state = batched_diffeqsolve(x0, times)
        return np.squeeze(state, axis=-2)

def sir_unnormalized_obj_func(states):
    if isinstance(states, np.ndarray):
        return states[..., 1:2] / (np.sum(states, -1, keepdims=True)) - 0.05 * (np.sum(states, -1, keepdims=True)) # states[..., -1:]
    else:
        return states[..., 1:2] / (tf.reduce_sum(states, -1, keepdims=True)) - 0.05 * (tf.reduce_sum(states, -1, keepdims=True)) # states[..., -1:]


@jit
def sird_observer(init_cond, times, t0, t1, problem_rng: random.PRNGKey, beta_range, gamma_range, mu_range, use_fixed_problem: bool = True, time_scaling: float = 1.0):
    if use_fixed_problem:
        beta = 1.0
        gamma = 0.5
        mu = 1.0
    else:  
        beta = random.uniform(problem_rng, (1,), beta_range[0], beta_range[1])[0]
        gamma = random.uniform(problem_rng, (1,), gamma_range[0], gamma_range[1])[0]
        mu = random.uniform(problem_rng, (1,), mu_range[0], mu_range[1])[0]

    times = times * time_scaling
    t1 = t1 * time_scaling

    x0 = init_cond

    def sir_dynamics(t, y, args):
        _S, _I, _R, _D = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        dSdt = -beta * _S * _I
        dIdt = beta * _S * _I - gamma * _I - mu * _I
        dRdt = gamma * _I
        dDdt = mu * _I
        return np.stack([dSdt, dIdt, dRdt, dDdt], axis=-1)

    if len(x0.shape) == 1:
        state = diffeqsolve(ODETerm(sir_dynamics), t0 = t0, t1 = t1, dt0 = None, 
                       stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), y0=x0, 
                       solver=Dopri5(), max_steps=1000, saveat=SaveAt(ts=times)).ys   
        return state
    else:
        batched_diffeqsolve = jax.vmap(
            lambda _x0, _times: diffeqsolve(
                ODETerm(sir_dynamics), 
                t0=t0, 
                t1=t1, 
                dt0=None, 
                stepsize_controller=PIDController(rtol=1e-7, atol=1e-9), 
                y0=_x0, 
                solver=Dopri5(), 
                max_steps=1000, 
                saveat=SaveAt(ts=_times)
            ).ys
        )
        state = batched_diffeqsolve(x0, times)
        return np.squeeze(state, axis=-2)

def sird_obj_func(states):
    if isinstance(states, np.ndarray):
        return states[..., 1:2] / (np.sum(states, -1, keepdims=True)) - 0.05 * (np.sum(states, -1, keepdims=True)) # states[..., -1:]
    else:
        return states[..., 1:2] / (tf.reduce_sum(states, -1, keepdims=True)) - 0.05 * (tf.reduce_sum(states, -1, keepdims=True)) # states[..., -1:]


@jit
def brusselator_observer(init_cond, times, t0, t1, problem_rng: random.PRNGKey, A_range, B_range, use_fixed_problem: bool = True, time_scaling: float = 1.0):
        if use_fixed_problem:
            A = np.array([0.8])
            B = np.array([1.5])
        else:  
            A = random.uniform(problem_rng, shape=(1,), minval=A_range[0], maxval=A_range[1])
            B = random.uniform(problem_rng, shape=(1,), minval=B_range[0], maxval=B_range[1])
        x0 = init_cond

        # times = times * 10 # use the same as in Loteka-Volterra
        times = times * time_scaling
        # t1 = t1 * 10 # use the same as in Loteka-Volterra
        t1 = t1 * time_scaling

        def dynamics(t, x, args):
            _x, _y = np.split(x, 2, axis=-1)
            _dx = A + (_x ** 2) * _y - _x * (B + 1)
            _dy = B * _x - (_x ** 2) * _y 
            return np.concatenate([_dx, _dy], axis=-1)

        if len(x0.shape) == 1:
            state = diffeqsolve(ODETerm(dynamics), t0 = t0, t1 = t1, dt0 = None, 
                           stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), y0=x0, 
                           solver=Dopri5(), max_steps=1000, saveat=SaveAt(ts=times)).ys   
            return state
        else:
            batched_diffeqsolve = jax.vmap(
                lambda _x0, _times: diffeqsolve(
                    ODETerm(dynamics), 
                    t0=t0, 
                    t1=t1, 
                    dt0=None, 
                    stepsize_controller=PIDController(rtol=1e-7, atol=1e-9), 
                    y0=_x0, 
                    solver=Dopri5(), 
                    max_steps=1000, 
                    saveat=SaveAt(ts=_times)
                ).ys
            )
            state = batched_diffeqsolve(x0, times)
            return np.squeeze(state, axis=-2)

def brusselator_obj_func(states):
    return states[..., -1:] # we use the last state!


@jax.jit
def selkov_observer(init_cond, times, t0, t1, problem_rng: random.PRNGKey, a_range, b_range, use_fixed_problem: bool = True, time_scaling: float = 1.0):
        if use_fixed_problem:
            a = np.array([0.25])
            b = np.array([0.45])
        else:  
            a = random.uniform(problem_rng, shape=(1,), minval=a_range[0], maxval=a_range[1])
            b = random.uniform(problem_rng, shape=(1,), minval=b_range[0], maxval=b_range[1])
        x0 = init_cond

        times = times * time_scaling
        t1 = t1 * time_scaling

        def dynamics(t, x, args):
            _x, _y = np.split(x, 2, axis=-1)
            _dx = -_x + a * _y + (_x ** 2) * _y
            _dy = b - a * _y - (_x ** 2) * _y
            return np.concatenate([_dx, _dy], axis=-1)

        if len(x0.shape) == 1:
            state = diffeqsolve(ODETerm(dynamics), t0 = t0, t1 = t1, dt0 = None, 
                           stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), y0=x0, 
                           solver=Dopri5(), max_steps=1000, saveat=SaveAt(ts=times)).ys   
            return state
        else:
            batched_diffeqsolve = jax.vmap(
                lambda _x0, _times: diffeqsolve(
                    ODETerm(dynamics), 
                    t0=t0, 
                    t1=t1, 
                    dt0=None, 
                    stepsize_controller=PIDController(rtol=1e-7, atol=1e-9), 
                    y0=_x0, 
                    solver=Dopri5(), 
                    max_steps=1000, 
                    saveat=SaveAt(ts=_times)
                ).ys
            )
            state = batched_diffeqsolve(x0, times)
            return np.squeeze(state, axis=-2)

def selkov_obj_func(states):
    return states[..., -1:] # we use the last state!


@jit
def react_net_observer(init_cond, 
                    times, 
                    t0, 
                    t1, 
                    problem_rng: random.PRNGKey,  
                    k0_1_range,
                    k0_2_range,
                    k0_3_range,
                    Ea_1_range,
                    Ea_2_range,
                    Ea_3_range,
                    T_range,
                    K1_range,
                    K2_range,
                    use_fixed_problem: bool = True, 
                    time_scaling: float = 1.0):

    if use_fixed_problem:
        k01 = 0.03
        k02 = 0.01
        k03 = 30
        Ea_f1 = 2000
        Ea_f2 = 1100
        Ea_f3 = 11000
        T = 100
        K1 = 2
        K2 = 2
    else:  
        k01_rng, k02_rng, k03_rng, Ea1_rng, Ea2_rng, Ea3_rng, T_rng = random.split(
            problem_rng, 7
        )
        k01 = random.uniform(k01_rng, shape=(1,), minval=k0_1_range[0], maxval=k0_1_range[1])
        k02 = random.uniform(k02_rng, shape=(1,), minval=k0_2_range[0], maxval=k0_2_range[1])
        k03 = random.uniform(k03_rng, shape=(1,), minval=k0_3_range[0], maxval=k0_3_range[1])
        Ea_f1 = random.uniform(Ea1_rng, shape=(1,), minval=Ea_1_range[0], maxval=Ea_1_range[1])
        Ea_f2 = random.uniform(Ea2_rng, shape=(1,), minval=Ea_2_range[0], maxval=Ea_2_range[1])
        Ea_f3 = random.uniform(Ea3_rng, shape=(1,), minval=Ea_3_range[0], maxval=Ea_3_range[1])
        T = random.uniform(T_rng, shape=(1,), minval=T_range[0], maxval=T_range[1])
        K1 = random.uniform(T_rng, shape=(1,), minval=K1_range[0], maxval=K1_range[1])
        K2 = random.uniform(T_rng, shape=(1,), minval=K2_range[0], maxval=K2_range[1])
    # x0 =  np.concatenat([init_cond, np.zeros(shape=(init_cond.shape[0], 1))], axis=-1)
    x0 = init_cond
    # zeros_to_add = np.zeros(x0.shape[:-1] + (1,))   

    # Apply padding with zeros
    # x0 = np.concatenate((x0, zeros_to_add), axis=-1)
    times = times * time_scaling
    t1 = t1 * time_scaling
    # times = times * 10 # use the same as in Loteka-Volterra
    # t1 = t1 * 10 # use the same as in Loteka-Volterra

    def dynamics(t, x, args):
        # State variables
        A, B, C, D = np.split(x, 4, axis=-1)  # Unpack the state vector
        # Constants
        R = 8.314  # J/(mol*K), universal gas constant

        # Parameters (example values, these need to be defined or estimated)
        k1_f = k01 * np.exp(- Ea_f1 / (R * T))
        k2_f = k02 * np.exp(- Ea_f2 / (R * T))
        r1 = k1_f * A * B - (k1_f / K1) * C
        r2 = k2_f * B * C - (k2_f / K2) * D
        r3 = k03 * np.exp(- Ea_f3 / (R * T)) * D

        # Rate of change of concentrations
        dAdt = - r1
        dBdt = -r1 - r2
        dCdt = r1 - r2
        dDdt = r2 - r3

        return np.concatenate([dAdt, dBdt, dCdt, dDdt], axis=-1)

    if len(x0.shape) == 1:
        state = diffeqsolve(ODETerm(dynamics), t0 = t0, t1 = t1, dt0 = None, 
                       stepsize_controller = PIDController(rtol=1e-7, atol=1e-9), y0=x0, 
                       solver=Dopri5(), max_steps=1000, saveat=SaveAt(ts=times)).ys   
        return state
    else:
        batched_diffeqsolve = jax.vmap(
            lambda _x0, _times: diffeqsolve(
                ODETerm(dynamics), 
                t0=t0, 
                t1=t1, 
                dt0=None, 
                stepsize_controller=PIDController(rtol=1e-7, atol=1e-9), 
                y0=_x0, 
                solver=Dopri5(), 
                max_steps=1000, 
                saveat=SaveAt(ts=_times)
            ).ys
        )
        state = batched_diffeqsolve(x0, times)
        return np.squeeze(state, axis=-2)


def react_net_obj(states):
    return states[..., -1:]