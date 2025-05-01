# Copyright 2020 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" This module contains implementations of various types of search space. """
from __future__ import annotations

from jax import random
from abc import ABC, abstractmethod
from typing import Optional, Sequence, TypeVar, overload

from jax import numpy as np
from jax.typing import ArrayLike

SearchSpaceType = TypeVar("SearchSpaceType", bound="SearchSpace")
""" A type variable bound to :class:`SearchSpace`. """


class SampleTimeoutError(Exception):
    """Raised when sampling from a search space has timed out."""


class SearchSpace(ABC):
    """
    A :class:`SearchSpace` represents the domain over which an objective function is optimized.
    """

    @abstractmethod
    def sample(self, num_samples: int, seed: Optional[int] = None) -> ArrayLike:
        """
        :param num_samples: The number of points to sample from this search space.
        :param seed: Random seed for reproducibility.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly from this search space.
        """

    def contains(self, value: ArrayLike) -> ArrayLike:
        """Method for checking membership.

        :param value: A point or points to check for membership of this :class:`SearchSpace`.
        :return: A boolean array showing membership for each point in value.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``value`` has a different
            dimensionality points from this :class:`SearchSpace`.
        """
        assert (
            len(value.shape) > 0 and value.shape[-1] == self.dimension
        ), f"""
            Dimensionality mismatch: space is {self.dimension}, value is {value.shape[-1]}
            """
        return self._contains(value)

    @abstractmethod
    def _contains(self, value: ArrayLike) -> ArrayLike:
        """Space-specific implementation of membership. Can assume valid input shape.

        :param value: A point or points to check for membership of this :class:`SearchSpace`.
        :return: A boolean array showing membership for each point in value.
        """

    def __contains__(self, value: ArrayLike) -> bool:
        """Method called by `in` operator. Doesn't support broadcasting as Python insists
        on converting the result to a boolean.

        :param value: A single point to check for membership of this :class:`SearchSpace`.
        :return: `True` if ``value`` is a member of this search space, else `False`.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``value`` has a different
            dimensionality from this :class:`SearchSpace`.
        """
        assert (
            len(value.shape) == 1
        ), f"""
            Rank mismatch: expected 1, got {len(value.shape)}. To get a tensor of boolean
            membership values from a tensor of points, use `space.contains(value)`
            rather than `value in space`.
            """
        return self.contains(value)

    @property
    @abstractmethod
    def dimension(self) -> ArrayLike:
        """The number of inputs in this search space."""

    @property
    @abstractmethod
    def lower(self) -> ArrayLike:
        """The lowest value taken by each search space dimension."""

    @property
    @abstractmethod
    def upper(self) -> ArrayLike:
        """The highest value taken by each search space dimension."""

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        :param other: A search space.
        :return: Whether the search space is identical to this one.
        """


class Box(SearchSpace):
    r"""
    Continuous :class:`SearchSpace` representing a :math:`D`-dimensional box in
    :math:`\mathbb{R}^D`. Mathematically it is equivalent to the Cartesian product of :math:`D`
    closed bounded intervals in :math:`\mathbb{R}`.
    """

    @overload
    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
    ):
        ...

    @overload
    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
    ):
        ...

    def __init__(
        self,
        lower: Sequence[float] | ArrayLike,
        upper: Sequence[float] | ArrayLike,
    ):
        r"""
        If ``lower`` and ``upper`` are `Sequence`\ s of floats (such as lists or tuples),
        they will be converted to tensors of dtype `DEFAULT_DTYPE`.

        :param lower: The lower (inclusive) bounds of the box. Must have shape [D] for positive D,
            and if a tensor, must have float type.
        :param upper: The upper (inclusive) bounds of the box. Must have shape [D] for positive D,
            and if a tensor, must have float type.
        :param constraints: Sequence of explicit input constraints for this search space.
        :param ctol: Tolerance to use to check constraints satisfaction.
        :raise ValueError (or tf.errors.InvalidArgumentError): If any of the following are true:

            - ``lower`` and ``upper`` have invalid shapes.
            - ``lower`` and ``upper`` do not have the same floating point type.
            - ``upper`` is not greater or equal to ``lower`` across all dimensions.
        """
        assert (
            len(lower.shape) == 1 and len(upper.shape) == 1
        ), "lower and upper must be 1D arrays"

        if isinstance(lower, Sequence):
            self._lower = np.array(lower)
            self._upper = np.array(upper)
        else:
            self._lower = np.asarray(lower)
            self._upper = np.asarray(upper)

        assert (
            self._lower.dtype == self._upper.dtype
        ), "lower and upper must have the same dtype"

        assert (
            np.all(self._lower <= self._upper)
        ), "upper must be greater or equal to lower across all dimensions"

        self._dimension = self._upper.shape[-1]

    def __repr__(self) -> str:
        """"""
        return f"Box({self._lower!r}, {self._upper!r}, {self._constraints!r}, {self._ctol!r})"

    @property
    def lower(self) -> ArrayLike:
        """The lower bounds of the box."""
        return self._lower

    @property
    def upper(self) -> ArrayLike:
        """The upper bounds of the box."""
        return self._upper

    @property
    def dimension(self) -> ArrayLike:
        """The number of inputs in this search space."""
        return self._dimension


    def _contains(self, value: ArrayLike) -> ArrayLike:
        return np.all(value >= self._lower, axis=-1) & np.all(
            value <= self._upper, axis=-1
        )

    def _sample(self, num_samples: int, seed: Optional[int] = None) -> ArrayLike:
        dim = self._lower.shape[-1]
        rng = random.PRNGKey(seed) if seed is not None else random.PRNGKey(0)
        return self._lower + (self._upper - self._lower) * random.uniform(
            rng, (num_samples, dim)
        )

    def sample(self, num_samples: int, seed: Optional[int] = None) -> ArrayLike:
        assert num_samples >= 0, "num_samples must be non-negative"
        return self._sample(num_samples, seed)

    def __eq__(self, other: object) -> bool:
        """
        :param other: A search space.
        :return: Whether the search space is identical to this one.
        """
        if not isinstance(other, Box):
            return NotImplemented
        return bool(
            np.all(self.lower == other.lower)
            and np.all(self.upper == other.upper)
        )
