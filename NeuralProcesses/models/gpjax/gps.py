"""
Modificatin of gps module from gpjax

The main modification is 
1. to add a custom jax based mean function
2. to add efficient posterior inference
"""

from gpjax.gps import Prior, ConjugatePosterior, Num, Array, Dataset, GaussianDistribution, cola, Gaussian, NonConjugatePosterior
from jax import numpy as jnp


class SupportNonFullCovConjugatePosterior(ConjugatePosterior):
    """
    Conjugate posteriors that support prediction without using full covariance matrices
    """
    def predict(
        self,
        test_inputs: Num[Array, "N D"],
        train_data: Dataset,
        full_cov: bool = False,
    ) -> GaussianDistribution:
        r"""Query the predictive posterior distribution.

        Conditional on a training data set, compute the GP's posterior
        predictive distribution for a given set of parameters. The returned function
        can be evaluated at a set of test inputs to compute the corresponding
        predictive density.

        The predictive distribution of a conjugate GP is given by
        $$
            p(\mathbf{f}^{\star}\mid \mathbf{y}) & = \int p(\mathbf{f}^{\star} \mathbf{f} \mid \mathbf{y})\\
            & =\mathcal{N}(\mathbf{f}^{\star} \boldsymbol{\mu}_{\mid \mathbf{y}}, \boldsymbol{\Sigma}_{\mid \mathbf{y}}
        $$
        where
        $$
            \boldsymbol{\mu}_{\mid \mathbf{y}} & = k(\mathbf{x}^{\star}, \mathbf{x})\left(k(\mathbf{x}, \mathbf{x}')+\sigma^2\mathbf{I}_n\right)^{-1}\mathbf{y}  \\
            \boldsymbol{\Sigma}_{\mid \mathbf{y}} & =k(\mathbf{x}^{\star}, \mathbf{x}^{\star\prime}) -k(\mathbf{x}^{\star}, \mathbf{x})\left( k(\mathbf{x}, \mathbf{x}') + \sigma^2\mathbf{I}_n \right)^{-1}k(\mathbf{x}, \mathbf{x}^{\star}).
        $$

        The conditioning set is a GPJax `Dataset` object, whilst predictions
        are made on a regular Jax array.

        Example:
            For a `posterior` distribution, the following code snippet will
            evaluate the predictive distribution.
            ```python
                >>> import gpjax as gpx
                >>> import jax.numpy as jnp
                >>>
                >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
                >>> ytrain = jnp.sin(xtrain)
                >>> D = gpx.Dataset(X=xtrain, y=ytrain)
                >>> xtest = jnp.linspace(0, 1).reshape(-1, 1)
                >>>
                >>> prior = gpx.gps.Prior(mean_function = gpx.mean_functions.Zero(), kernel = gpx.kernels.RBF())
                >>> posterior = prior * gpx.likelihoods.Gaussian(num_datapoints = D.n)
                >>> predictive_dist = posterior(xtest, D)
            ```

        Args:
            test_inputs (Num[Array, "N D"]): A Jax array of test inputs at which the
                predictive distribution is evaluated.
            train_data (Dataset): A `gpx.Dataset` object that contains the input and
                output data used for training dataset.
            full_cov (bool, optional): Whether to return the full covariance matrix

        Returns
        -------
            GaussianDistribution: A function that accepts an input array and
                returns the predictive distribution as a `GaussianDistribution`.
        """
        # Unpack training data
        x, y = train_data.X, train_data.y

        # Unpack test inputs
        t = test_inputs

        # Observation noise o²
        obs_noise = self.likelihood.obs_stddev**2
        mx = self.prior.mean_function(x)

        # Precompute Gram matrix, Kxx, at training inputs, x
        Kxx = self.prior.kernel.gram(x)
        Kxx += cola.ops.I_like(Kxx) * self.jitter

        # Σ = Kxx + Io²
        Sigma = Kxx + cola.ops.I_like(Kxx) * obs_noise
        Sigma = cola.PSD(Sigma)

        mean_t = self.prior.mean_function(t)
        if full_cov:
            Ktt = self.prior.kernel.gram(t)
        else:
            Ktt = self.prior.kernel.diagonal(t)
        Kxt = self.prior.kernel.cross_covariance(x, t) 
        Sigma_inv_Kxt = cola.solve(Sigma, Kxt)

        # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
        mean = mean_t + jnp.matmul(Sigma_inv_Kxt.T, y - mx)

        # Ktt  -  Ktx (Kxx + Io²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        covariance += cola.ops.I_like(covariance) * self.prior.jitter
        covariance = cola.PSD(covariance)

        return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)



class SupportNonFullCovPriors(Prior):
    """
    Priors that support prediction without using full covariance matrices
    """
    def __mul__(self, other):
        r"""Combine the prior with a likelihood to form a posterior distribution.

        The product of a prior and likelihood is proportional to the posterior
        distribution. By computing the product of a GP prior and a likelihood
        object, a posterior GP object will be returned. Mathematically, this can
        be described by:
        ```math
        p(f(\cdot) \mid y) \propto p(y \mid f(\cdot))p(f(\cdot)),
        ```
        where $`p(y | f(\cdot))`$ is the likelihood and $`p(f(\cdot))`$ is the prior.

        Example:
        ```python
            >>> import gpjax as gpx
            >>>
            >>> meanf = gpx.mean_functions.Zero()
            >>> kernel = gpx.kernels.RBF()
            >>> prior = gpx.gps.Prior(mean_function=meanf, kernel = kernel)
            >>> likelihood = gpx.likelihoods.Gaussian(num_datapoints=100)
            >>>
            >>> prior * likelihood
        ```
        Args:
            other (Likelihood): The likelihood distribution of the observed dataset.

        Returns
        -------
            Posterior: The relevant GP posterior for the given prior and
                likelihood. Special cases are accounted for where the model
                is conjugate.
        """
        return construct_posterior(prior=self, likelihood=other)


def construct_posterior(prior, likelihood):
    r"""Utility function for constructing a posterior object from a prior and
    likelihood. The function will automatically select the correct posterior
    object based on the likelihood.

    Args:
        prior (Prior): The Prior distribution.
        likelihood (AbstractLikelihood): The likelihood that represents our
            beliefs around the distribution of the data.

    Returns
    -------
        AbstractPosterior: A posterior distribution. If the likelihood is
            Gaussian, then a `ConjugatePosterior` will be returned. Otherwise,
            a `NonConjugatePosterior` will be returned.
    """
    if isinstance(likelihood, Gaussian):
        return SupportNonFullCovConjugatePosterior(prior=prior, likelihood=likelihood)

    return NonConjugatePosterior(prior=prior, likelihood=likelihood)
