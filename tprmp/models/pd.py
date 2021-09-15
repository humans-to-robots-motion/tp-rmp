import numpy as np
import logging


class PDController(object):
    '''
    Simple PD controller
    '''
    logger = logging.getLogger(__name__)

    def __init__(self, manifold, **kwargs):
        self.manifold = manifold
        self.Kp = kwargs.get('Kp', np.eye(self.manifold.dim_T))
        self.Kd = kwargs.get('Kd', 0.1 * np.eye(self.manifold.dim_T))
        self.xd = None
        self.dxd = None

    def update_targets(self, xd, dxd):
        assert xd.shape[0] == self.manifold.dim_M
        assert dxd.shape[0] == self.manifold.dim_T
        self.xd = xd
        self.dxd = dxd

    def retrieve(self, x, dx):
        return -(self.Kp @ self.manifold.log_map(x, base=self.xd) + self.Kd @ (dx - self.dxd))
