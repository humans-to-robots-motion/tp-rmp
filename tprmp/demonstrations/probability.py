import numpy as np


class ManifoldGaussian(object):
    """
    Gaussian defined on manifold.
    "Programming by Demonstration on Riemannian Manifolds", M.J.A. Zeestraten, 2018.
    """

    def __init__(self, manifold, mean, cov):
        self._manifold = manifold
        if mean.shape != (manifold.dim_M,):
            raise TypeError(f'[Manifold]: mean dim_M {mean.shape[0]} and the manifold dim_M {manifold.dim_M} mismatch!')
        if cov.shape != (manifold.dim_T, manifold.dim_T):
            raise TypeError(f'[Manifold]: cov shape {cov.shape} and the manifold dim_T {manifold.dim_T, manifold.dim_T} mismatch!')
        self.mean = mean
        self.cov = cov

    def pdf(self, p):
        """
        Evaluates pdf of the distribution at the point p.

        Parameters
        ----------
        :param p: np.array of shape (self.manifold.dim_M,) or (self.manifold.dim_M, t)

        Returns
        -------
        :return pdf_p: float, or np.array of shape (t,), the pdf values at the points p.
        """
        v = self.manifold.log_map(p, base=self.mean)
        return self._nf * np.exp(-(v * self.cov_inv.dot(v)).sum(0) / 2)

    def transform(self, A, b):
        """
        Gaussian transformation. Note that A is of shape (dim_T, dim_T) while b is of
        shape (dim_M).

        :param A: np.array of shape (dim_T, dim_T), rotation in tangent space
        :param b: np.array of shape (dim_M,), translation in manifold space

        Returns
        -------
        :return ManifoldGaussian, the transformed Gaussian
        """
        if A.shape != (self.manifold.dim_T, self.manifold.dim_T):
            raise RuntimeError('[Manifold]: Expected A to be of the dimension of the tangent'
                               ' space (%s, %s) and not %s' % (self.manifold.dim_T, self.manifold.dim_T, A.shape))
        if b.shape != (self.manifold.dim_M,):
            raise RuntimeError('[Manifold]: Expected b to be of the dimension of the manifold'
                               ' space (%s,) and not %s' % (self.manifold.dim_M, b.shape))
        mu_trafo = self.manifold.exp_map(A.dot(self.manifold.log_map(self.mean)), base=b)  # (2.53)
        sigma_trafo = self.manifold.matrix_parallel_transport(A.dot(self.cov).dot(A.T), b, mu_trafo)  # (2.54)
        return ManifoldGaussian(self.manifold, mu_trafo, sigma_trafo)

    def kl_divergence_mvn(self, g):
        term1 = np.log(np.linalg.det(g.cov)) - np.log(np.linalg.det(self.cov))
        term2 = np.trace(np.linalg.solve(g.cov, self.cov)) - self.manifold.dim_T
        v = g.manifold.log_map(self.mean, base=g.mean)
        term3 = g.cov_inv.dot(v).dot(v)
        return (term1 + term2 + term3) / 2

    def recompute_inv(self):
        self._cov_inv = np.linalg.inv(self.cov)
        self._nf = 1 / (np.sqrt((2 * np.pi)**self.manifold.dim_T * np.linalg.det(self.cov)))

    @property
    def manifold(self):
        return self._manifold

    @property
    def cov(self):
        return self._cov

    @property
    def nf(self):
        return self._nf

    @cov.setter
    def cov(self, value):
        self._cov = value
        self.recompute_inv()

    @property
    def cov_inv(self):
        return self._cov_inv
