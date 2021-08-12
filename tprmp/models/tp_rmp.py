import os
import logging
import numpy as np
import pickle
import time

from tprmp.models.tp_hsmm import TPHSMM
from tprmp.models.rmp import compute_policy
from tprmp.optimizer.dynamics import optimize_dynamics


_path_file = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(_path_file, '..', '..', 'data', 'tasks')


class TPRMP(object):
    '''
    Wrapper of TPHSMM to retrieve RMP.
    '''
    logger = logging.getLogger(__name__)

    def __init__(self, **kwargs):
        self._model = TPHSMM(**kwargs)
        self._phi0 = None
        self._d0 = None

    def save(self, name=None):
        self.model.save(name)
        file = os.path.join(DATA_PATH, self.model.name, 'models', name if name is not None else ('dynamics_' + str(time.time()) + '.p'))
        os.makedirs(file, exist_ok=True)
        with open(file, 'wb') as f:
            pickle.dump(self.parameters(), f)

    def retrieve(self, frames, **kwargs):
        """
        Retrieve global RMP.
        """

    def compute_global_policy(self, x, dx, frames):  # TODO: add Riemmanian metric
        if not set(self.model.frame_names).issubset(set(frames)):
            raise IndexError(f'[TPRMP]: Frames must be subset of {self.model.frame_names}')
        policies = np.zeros((len(frames), self.model.manifold.dim_T))
        for i, f_key in enumerate(self.model.frame_names):
            # compute local policy
            lx = frames[f_key].pullback(x)
            ldx = frames[f_key].pullback_tangent(dx)
            local_policy = compute_policy(self._phi0[f_key], self._d0[f_key], lx, ldx, self.model.get_local_gmm(f_key))
            policies[i] = frames[f_key].transform_tangent(local_policy)
        return policies.sum(axis=0)  # TODO: may need weighted sum based on frames

    def train(self, demos, **kwargs):
        """
        Trains the TP-RMP with a given set of demonstrations.

        Parameters
        ----------
        :param demos: list of Demonstration objects
        """
        alpha = kwargs.get('alpha', 0.5)
        # train TP-HSMM/TP-GMM
        self.model.train(demos, **kwargs)
        # train dynamics
        self._phi0, self._d0 = optimize_dynamics(self.model, demos, alpha)

    @staticmethod
    def load(task_name, model_name='sample.p'):
        """
        Parameters
        ----------
        :param model_name: name of model in data/models
        """
        tprmp = TPRMP()
        tprmp._model = TPHSMM.load(task_name, model_name)
        return tprmp

    @property
    def name(self):
        return self._model.name

    @property
    def model(self):
        return self._model

    @property
    def task_parameters(self):
        return self._model.frame_names

    @property
    def dt(self):
        return self._model.dt
