import numpy as np
import logging

from tprmp.demonstrations.manifold import Manifold

logger = logging.getLogger(__name__)


class RMPNode:
    def __init__(self, name, parent=None, manifold=None, psi=None, J=None, J_dot=None):
        self.name = name
        self.parent = parent
        self.children = []
        # connect the node to its parent
        if self.parent:
            self.parent.add_child(self)
        # mapping/J/J_dot for the edge from the parent to the node
        self.psi = psi
        self.J = J
        self.J_dot = J_dot
        # state
        self.x = None
        self.dx = None
        # RMP
        self.f = None
        self.a = None
        self.M = None
        # space structure
        self.manifold = manifold

    def add_child(self, child):
        self.children.append(child)

    def update_jacobian(self, J, J_dot=None):
        self.J = J
        self.J_dot = J_dot

    def pushforward(self):
        logger.debug(f'{self.name}: pushforward')
        if self.psi is not None and self.parent.x is not None:
            self.x = self.psi(self.parent.x)
        if self.J is not None and self.parent.dx is not None:
            self.dx = np.dot(self.J(self.parent.x), self.parent.dx)
        for child in self.children:
            child.pushforward()

    def pullback(self):
        for child in self.children:
            child.pullback()
        logger.debug(f'{self.name}: pullback')
        if self.manifold is None:
            self.manifold = Manifold.get_euclidean_manifold(self.x.shape[0])
        f = np.zeros(self.manifold.dim_T)
        M = np.eye(self.manifold.dim_T)
        for child in self.children:
            if child.f is not None and child.M is not None:
                f_term = child.f
                if child.J_dot is not None:
                    f_term -= np.dot(np.dot(child.M, child.J_dot(self.x, self.dx)), self.dx)
                J = child.J(self.x)
                f += np.dot(J.T, f_term)
                M += np.dot(np.dot(J.T, child.M), J)
        self.f = f
        self.M = M


class RMPRoot(RMPNode):
    def __init__(self, name, manifold=None):
        RMPNode.__init__(self, name, manifold=manifold)

    def set_root_state(self, x, dx):
        self.x = x
        self.dx = dx

    def pushforward(self):
        logger.debug(f'{self.name}: Root pushforward')
        for child in self.children:
            child.pushforward()

    def resolve(self):
        logger.debug(f'{self.name}: Root pullback')
        self.a = np.dot(np.linalg.pinv(self.M), self.f)
        return self.a

    def solve(self, x, dx):
        self.set_root_state(x, dx)
        self.pushforward()
        self.pullback()
        return self.resolve()


class RMPLeaf(RMPNode):
    def __init__(self, name, rmp_func, parent=None, manifold=None, psi=None, J=None, J_dot=None):
        RMPNode.__init__(self, name, parent, manifold, psi, J, J_dot)
        self.rmp_func = rmp_func

    def pullback(self):
        logger.debug(f'{self.name}: leaf pullback')
        self.M, self.f = self.rmp_func(self.x, self.dx)
