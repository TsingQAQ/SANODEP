import jax
import math
import warnings
from abc import ABC
from jax import numpy as np
from typing import Optional
from flax.linen import Module, Embed
from flax.linen.normalization import Array
from flax.linen.dtypes import promote_dtype


class PositionalEncoding(Module, ABC):
    """
    Positional Encoding 
    """


class SEFTTimeEncoding(PositionalEncoding):
    dimensionality: int
    """
    Time Encoding from paper Set Functions for Time Series
        This is NOTHING different than the positional encoding used in Transformer
    """
    def setup(self):
        assert self.dimensionality % 2 == 0, ValueError(f'SEFT Encoding only support even number of embedding dimensionalities but got {self.dimensionality}')
        self._dim = self.dimensionality

    def __call__(self, times: jax.typing.ArrayLike, max_period: Optional[int] = 10000) -> jax.typing.ArrayLike:
        """
        :param times: (batch) time steps to be embedded [...]
        :param maximum sequence length
        :return: [..., embedding_dim]
        """
        # times = np.expand_dims(times, -1) # [Batch_dim, 1]
        half_dim = self._dim // 2
        emb = max_period ** (2 * (np.arange(half_dim)) / self._dim) # [d_model // 2]
        emb = times / emb[None, :] # [Batch_dim, d_model // 2]
        emb = np.concatenate([np.expand_dims(np.sin(emb), -1), 
                              np.expand_dims(np.cos(emb), -1)], axis=-1) # [Batch_dim, d_model // 2 , 2]
        emb = emb.reshape(emb.shape[:-2] + (-1,)) # [..., d_model] (sin, cos, sin, cos ...)
        return emb


