# from functools import partial
# import operator
# import warnings
# import numpy as np
from typing import Any, Optional, Tuple, Union
# 
# import jax
# import jax.numpy as jnp
# from jax import custom_jvp
# from jax import lax
# from jax._src import core
# from jax._src import dtypes
# from jax._src import util
# from jax._src.core import AxisName
# from jax._src.ops.special import logsumexp as _logsumexp
# 
Array = Any
# 
# import builtins
# from functools import partial
# import math
# import operator
# from typing import (
#     overload, Any, Callable, Literal, Optional, Protocol, Sequence, Tuple, Union)
# import warnings
# 
# import numpy as np
# 
# from jax import lax
# from jax._src import api
# from jax._src import core
# from jax._src import dtypes
# from jax._src.numpy import ufuncs
# from jax._src.numpy.util import (
#     _broadcast_to, check_arraylike, _complex_elem_type,
#     promote_dtypes_inexact, promote_dtypes_numeric, _where, _wraps)
# from jax._src.lax import lax as lax_internal
# from jax._src.typing import Array, ArrayLike, DType, DTypeLike
# from jax._src.util import (
#     canonicalize_axis as _canonicalize_axis, maybe_named_axis)
# 
# 
# _all = builtins.all
# _lax_const = lax_internal._const
# 
# 
# Axis = Union[None, int, Sequence[int]]
# from jax._src.numpy.reductions import _ensure_optional_axes, _reduction

# @_wraps(np.max, skip_params=['out'])
# def _max(a: ArrayLike, axis: Axis = None, out: None = None,
#         keepdims: bool = False, initial: Optional[ArrayLike] = None,
#         where: Optional[ArrayLike] = None) -> Array:
#   return _reduce_max(a, axis=_ensure_optional_axes(axis), out=out,
#                      keepdims=keepdims, initial=initial, where=where)
# 
# @partial(api.jit, static_argnames=('axis', 'keepdims'), inline=True)
# def _reduce_max(a: ArrayLike, axis: Axis = None, out: None = None,
#                 keepdims: bool = False, initial: Optional[ArrayLike] = None,
#                 where: Optional[ArrayLike] = None) -> Array:
#   return _reduction(a, "max", np.max, lax.max, -np.inf, has_identity=False,
#                     axis=axis, out=out, keepdims=keepdims,
#                     initial=initial, where_=where, parallel_reduce=lax.pmax)

from flax.linen import softmax as builtin_softmax
from jax import numpy as np

def self_implemented_softmax(x, where):
   exponentials = np.exp(x)
   denominator = np.sum(exponentials * where, axis=-1, keepdims=True)
   return exponentials / denominator


def softmax(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None,
            initial: Optional[Array] = None) -> Array:
    r"""Softmax function.

    Computes the function which rescales elements to the range :math:`[0, 1]`
    such that the elements along :code:`axis` sum to :math:`1`.

    .. math ::
      \mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    Args:
      x : input array
      axis: the axis or axes along which the softmax should be computed. The
        softmax output summed across these dimensions should sum to :math:`1`.
        Either an integer or a tuple of integers.
      where: Elements to include in the :code:`softmax`.
      initial: The minimum value used to shift the input array. Must be present
        when :code:`where` is not None.
    """
    debug = builtin_softmax(x, axis, where, initial)
    return self_implemented_softmax(x, where)
    # return _softmax_deprecated(x, axis, where, initial)