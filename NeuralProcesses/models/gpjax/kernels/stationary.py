from gpjax.kernels.stationary.matern32 import Union, ScalarFloat, Float, param_field, \
    Array, jnp, tfb, tfd, euclidean_distance, build_student_t_distribution, dataclass
from .base import AbstractKernelSupportDiagonalCov


@dataclass
class Matern32(AbstractKernelSupportDiagonalCov):
    r"""The Matérn kernel with smoothness parameter fixed at 1.5."""

    lengthscale: Union[ScalarFloat, Float[Array, " D"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus()
    )
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    name: str = "Matérn32"

    def __call__(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
    ) -> ScalarFloat:
        r"""Compute the Matérn 3/2 kernel between a pair of arrays.

        Evaluate the kernel on a pair of inputs $`(x, y)`$ with
        lengthscale parameter $`\ell`$ and variance $`\sigma^2`$.

        ```math
            k(x, y) = \sigma^2 \exp \Bigg(1+ \frac{\sqrt{3}\lvert x-y \rvert}{\ell^2}  \Bigg)\exp\Bigg(-\frac{\sqrt{3}\lvert x-y\rvert}{\ell^2} \Bigg)
        ```

        Args:
            x (Float[Array, " D"]): The left hand argument of the kernel function's call.
            y (Float[Array, " D"]): The right hand argument of the kernel function's call.

        Returns
        -------
            ScalarFloat: The value of $k(x, y)$.
        """
        x = self.slice_input(x) / self.lengthscale
        y = self.slice_input(y) / self.lengthscale
        tau = euclidean_distance(x, y)
        K = self.variance * (1.0 + jnp.sqrt(3.0) * tau) * jnp.exp(-jnp.sqrt(3.0) * tau)
        return K.squeeze()

    @property
    def spectral_density(self) -> tfd.Distribution:
        return build_student_t_distribution(nu=3)