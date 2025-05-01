"""
Used to transform some parameters to another space for optimization, 
this is used to make sure some parameters may satisfy some certain constraints like non-negativity 
"""
from abc import ABC
from functools import partial
from jax import numpy as np
from jax import jit



@partial(jit, static_argnames=['transform']) 
def transform_diagonal(matrix, transform):
    # Get the indices of the diagonal
    diag_indices = np.diag_indices_from(matrix)
    # Extract the diagonal elements
    diagonal = matrix[diag_indices]
    # Apply the transformation to the diagonal elements
    transformed_diagonal = transform(diagonal)
    # Create a new matrix with the transformed diagonal
    transformed_matrix = matrix.at[diag_indices].set(transformed_diagonal)
    return transformed_matrix


class Transformation(ABC):
    """
    Parameter transformation
    """
    @classmethod
    def forward(self, x):
        """
        transform the raw input to scale input
        """

    @classmethod
    def backward(self, x):
        """
        inverse the transformation
        """

    def __call__(self, x):
        return self.forward(x)


class IdentityTransform(Transformation):
    @staticmethod
    def forward(x):
        return x
    
    @staticmethod
    def backward(x):
        return x
        

if __name__ == '__main__':
    from jax import numpy as np
    print(transform_diagonal(-np.ones(shape=(3 ,3)), lambda xs: np.exp(xs)))