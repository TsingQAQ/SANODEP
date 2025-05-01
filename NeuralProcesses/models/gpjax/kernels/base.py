from gpjax.kernels.base import AbstractKernel, Num, Array


class AbstractKernelSupportDiagonalCov(AbstractKernel):
    def diagonal(self, x: Num[Array, "N D"]):
        return self.compute_engine.diagonal(self, x)