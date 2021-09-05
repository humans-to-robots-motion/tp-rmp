from tprmp.demonstrations.manifold import Manifold
from numpy.linalg import norm
import numpy as np
import logging

from tprmp.models.rmp_tree import RMPLeaf

logger = logging.getLogger(__name__)


class CollisionAvoidance(RMPLeaf):
    """
    Obstacle avoidance RMP leaf.
    This class considers a simple round obstacle. We can easily extend collision avoidance to Signed Distance Map if needed!
    """
    def __init__(self, name, parent, c=np.zeros(3), R=1., epsilon=0.2, alpha=1e-12, gamma=1e-5, eta=0):
        self.R = R
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.c = c

        def rmp_func(x, dx, M_limit=1e5, f_limit=1e10):
            if x < 0:
                w = 1e10
                grad_w = 0
            else:
                w = self.gamma / x ** 4
                grad_w = -self.gamma * 4.0 / x ** 5
            u = epsilon + self.gamma * min(0, dx) * dx
            grad_u = self.gamma * 2 * min(0, dx)
            g = w * u
            grad_Phi = alpha * w * grad_w
            xi = 0.5 * dx ** 2 * u * grad_w
            M = g + 0.5 * dx * w * grad_u
            M = min(max(M, self.epsilon), M_limit)
            Bdx = eta * g * dx
            f = - grad_Phi - xi - Bdx
            f = min(max(f, -f_limit), f_limit)
            return M, f
        RMPLeaf.__init__(self, name, rmp_func, parent=parent, manifold=Manifold.get_euclidean_manifold(1))
        # mappings f: R^3 -> R
        self.psi = lambda x: np.array(norm(x - self.c) / self.R - 1)  # noqa
        self.J = lambda x: (1. / (self.R * norm(x - self.c))) * (x - self.c).T  # noqa
        self.J_dot = lambda x, dx: ((-1. / norm(x - self.c) ** 3) * np.outer((x - self.c), (x - self.c)) + (1. / norm(x - self.c)) * np.eye(3)) @ dx / self.R  # noqa
