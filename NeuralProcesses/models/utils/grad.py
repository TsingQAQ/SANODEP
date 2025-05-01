"""
Custom implemented jacobian calculation from jax repository https://github.com/google/jax/pull/762
"""

import functools
import jax
import jax.numpy as np
from jax import jvp

def value_and_jacfwd(f, x):
  pushfwd = functools.partial(jax.jvp, f, (x,))
  basis = np.eye(x.size, dtype=x.dtype)
  y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
  return y, jac

def value_and_jacrev(f, x):
  y, pullback = jax.vjp(f, x)
  basis = np.eye(y.size, dtype=y.dtype)
  jac = jax.vmap(pullback)(basis)
  return y, jac


def value_and_jvp(fun, argnums=0, has_aux=False):
    def wrapped(*args, **kwargs):
        if not isinstance(argnums, tuple):
            argnums = (argnums,)

        # Prepare the seed for jvp
        seeds = tuple(np.ones_like(args[i]) if i in argnums else np.zeros_like(args[i]) for i in range(len(args)))

        if has_aux:
            fun_value, vjp_fun, aux = jvp(lambda *args: fun(*args, **kwargs), args, seeds)
            return (fun_value, vjp_fun), aux
        else:
            fun_value, vjp_fun = jvp(lambda *args: fun(*args, **kwargs), args, seeds)
            return fun_value, vjp_fun

    return wrapped

if __name__ == '__main__':
    # Define a simple function for testing
    def f(x):
        return x ** 2

    x = np.array([1.0, 2.0, 3.0])

    # Test value_and_jacfwd
    y, jac = value_and_jacfwd(f, x)
    print("Forward mode:")
    print("y:", y)
    print("jac:", jac)

    # Test value_and_jacrev
    y, jac = value_and_jacrev(f, x)
    print("Reverse mode:")
    print("y:", y)
    print("jac:", jac)