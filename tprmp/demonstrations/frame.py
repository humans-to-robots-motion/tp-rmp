import numpy as np
import logging

from tprmp.demonstrations.manifold import Manifold
logger = logging.getLogger(__name__)


class Frame(object):
    def __init__(self, A, b, manifold=None):
        """
        Candidate frame given by linear map with Jacobians A and displacement b.

        :param A: np.array of shape (dim_T, dim_T), rotation in tangent space
        :param b: np.array of shape (dim_M,), translation in manifold space
        """
        if A.shape[1] != A.shape[0]:
            raise RuntimeError("Expected A to be a square matrix and not %s." % A.shape)
        self._A = A
        self._A_inv = None
        if manifold is None:
            manifold = Manifold.get_euclidean_manifold(b.shape[0])
        if manifold.dim_T != A.shape[0] or manifold.dim_M != b.shape[0]:
            raise RuntimeError("Expected a manifold with dim_T = %s and dim_M = %s to match the dimensions of A"
                               " and b." % (A.shape[0], b.shape[0]))
        self._b = b
        self._manifold = manifold

    def transform(self, x):
        """
        Transforms point x from this frame to global frame.

        Parameters
        ----------
        :param x: np.array of shape (dim_M, t)

        Returns
        -------
        :return xi: np.array of shape (dim_M, t)
        """
        return self._manifold.exp_map(self.A.dot(self._manifold.log_map(x)), base=self.b)

    def pullback(self, xi):
        """
        Transforms point xi from global frame to this frame.

        Parameters
        ----------
        :param xi: np.array of shape (dim_M, t)

        Returns
        -------
        :return x: np.array of shape (dim_M, t)
        """
        return self._manifold.exp_map(self.A_inv.dot(self._manifold.log_map(xi, base=self.b)))

    def transform_tangent(self, x):
        """
        Transforms tangent vectors x from this frame to global frame.  # NOTE: only work with static frame

        Parameters
        ----------
        :param x: np.array of shape (dim_T, t)

        Returns
        -------
        :return xi: np.array of shape (dim_T, t)
        """
        return self.A.dot(x)

    def pullback_tangent(self, xi):
        """
        Transforms tangent vectors xi from global frame to this frame.  # NOTE: only work with static frame

        Parameters
        ----------
        :param xi: np.array of shape (dim_T, t)

        Returns
        -------
        :return x: np.array of shape (dim_T, t)
        """
        return self.A_inv.dot(xi)

    @property
    def A(self):
        return self._A

    @property
    def A_inv(self):
        if self._A_inv is None:
            self._A_inv = np.linalg.inv(self._A)
        return self._A_inv

    @property
    def b(self):
        return self._b

    @property
    def manifold(self):
        return self._manifold
