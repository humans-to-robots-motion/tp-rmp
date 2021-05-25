import numpy as np


class Frame(object):
    def __init__(self, A, b):
        """
        Candidate frame given by orientation A and origin b.

        :param A: np.array of shape (dim_T, dim_T), rotation in tangent space
        :param b: np.array of shape (dim_M,), translation in manifold space
        TODO: Add manifolds framing
        """
        if A.shape[1] != A.shape[0]:
            raise RuntimeError("Expected A to be a square matrix and not %s." % A.shape)
        self._A = A
        self._A_inv = None
        self._b = b

    def transform(self, x):
        """
        Transforms point x from this frame to global frame.

        Parameters
        ----------
        :param x: np.array of shape (dim_M,)

        Returns
        -------
        :return xi: np.array of shape (dim_M,)
        """
        return self.A.dot(x) + self.b

    def pullback(self, xi):
        """
        Transforms point xi from global frame to this frame.

        Parameters
        ----------
        :param xi: np.array of shape (dim_M,)

        Returns
        -------
        :return x: np.array of shape (dim_M,)
        """
        return self.A_inv.dot(xi - self.b)

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
