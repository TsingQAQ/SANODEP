import diffrax
from diffrax import diffeqsolve, ODETerm, Euler


def solve_ode(term, initial_x, t0, t1, ts, args, dt0, solver='Euler'):
    """
    A wrapper for solving ODE using diffrax
    """
    solution = diffeqsolve(term, getattr(diffrax, solver)(), t0=t0, t1=t1, dt0=dt0, 
                           y0=initial_x, args = args, saveat=diffrax.SaveAt(ts=ts))
    return solution.ys