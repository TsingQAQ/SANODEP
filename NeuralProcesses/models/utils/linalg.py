from jax import jit
from jax import numpy as np
from functools import partial


def calc_elmnt_num_in_ltri_from_dim(dim: int, diagonal_included: bool = True) -> int:
    """
    Given a 1-d vector, or a specific matrix dimenisonality, calculating the element number of lower triangular matrix ()

    :param dim dimensionality of the matrix
    """
    if diagonal_included:
        return int(dim * (dim + 1) / 2)
    else:
        return int(dim * (dim - 1) / 2)


@partial(jit, static_argnames=['dim']) 
def create_lower_tri_matrix(vector, dim):
    """
    create a lower triangular matrix given a 1-d vector

    here, we utilize the Log-Cholesky parametrization 
    refer Sec 2.2. of pinheiro1996unconstrained

    note that this lower triangular matrix does not satisfy the conditional of Cholesky decomposition yet 
    since its digonal element can be negative, to use it as a Cholesky decomposed lower triangular matrix, 
    we parameterize vector as it is, but assume the diagonal part has been 
    parameterized in log scale, so to the end we add a exponential transformation
    on diagonal elements, similar as 
    https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/bijectors/TransformDiagonal

    this exponential transformation need to be called separately from transofrmation.py
    """
    # Calculate the size of the matrix
    # dim = ((np.sqrt(1 + 8 * len(vector)) - 1) / 2).astype(np.int32) 
    # Create an empty square matrix of the calculated size
    tri_matrix = np.zeros((dim, dim))

    # Get the indices for the lower triangle including the diagonal
    indices = np.tril_indices(dim)

    # Assign the values from the array to the lower triangular part of the matrix
    tri_matrix = tri_matrix.at[indices].set(vector)

    return tri_matrix


if __name__ == '__main__':
    # create a lower triangular matrix from an 1d array 
    from jax import numpy as np
    xs = np.arange(calc_elmnt_num_in_ltri_from_dim(dim=3))[::-1]
    tri_xs = create_lower_tri_matrix(xs, dim=3)
    print(tri_xs)

    xs = np.arange(calc_elmnt_num_in_ltri_from_dim(dim=10))[::-1]
    tri_xs = create_lower_tri_matrix(xs, dim=10)
    print(tri_xs)