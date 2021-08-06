import os
import logging
import numpy as np

from tprmp.models.tp_hsmm import TPHSMM


_path_file = os.path.dirname(os.path.realpath(__file__))


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

    def retrieve(self, frames, **kwargs):
        """
        Retrieve global RMP.
        """

    def compute_canonical_rmp(self, x, t, global_mvns, **kwargs):
        manifold = global_mvns[0].manifold
        K = kwargs.get('K', 10 * np.eye(manifold.dim_T))
        hard = kwargs.get('hard', True)
        weights = self.compute_force_weights(x, t, global_mvns, **kwargs)
        if hard:
            return K @ manifold.log_map(x, base=global_mvns[np.argmax(weights)].mean)
        else:
            phi = np.zeros((self.model.num_comp, manifold.dim_T))
            for k in range(self.model.num_comp):
                phi[k] = K @ manifold.log_map(x, base=global_mvns[k].mean)
            return weights @ phi

    def compute_policy(self, x, x_dot, global_mvns):
        phi = self.compute_potentials(x, global_mvns)
        weights = TPRMP.compute_obsrv_prob(x, global_mvns)
        Phi = weights @ phi
        num_comp = len(global_mvns)
        manifold = global_mvns[0].manifold
        Fs = np.zeros((num_comp, manifold.dim_T))
        for k in range(num_comp):
            Fs[k] = weights[k] * (phi[k] - Phi) * global_mvns[k].cov_inv @ manifold.log_map(x, base=global_mvns[k].mean)
            Fs[k] += -weights[k] * global_mvns[k].cov_inv @ manifold.log_map(x, base=global_mvns[k].mean)
            Fs[k] += -weights[k] * self._d0[k] * x_dot
        return Fs.sum(axis=0)

    def compute_potentials(self, x, global_mvns):
        num_comp = len(global_mvns)
        phi = np.zeros(num_comp)
        manifold = global_mvns[0].manifold
        for k in range(num_comp):
            comp = global_mvns[k]
            v = manifold.log_map(x, base=comp.mean)
            phi[k] = self._phi0[k] + v.T @ comp.cov_inv @ v
        return phi

    def train(self, demos, **kwargs):
        """
        Trains the TP-RMP with a given set of demonstrations.

        Parameters
        ----------
        :param demos: list of Demonstration objects
        """
        self.model.train(demos, **kwargs)

    @staticmethod
    def compute_obsrv_prob(obsrv, global_mvns, normalized=True):
        """
        Parameters
        ----------
        :param model_name: name of model in data/models
        """
        num_comp = len(global_mvns)
        prob = np.zeros(num_comp)
        for k in range(num_comp):
            prob[k] = global_mvns[k].pdf(obsrv)
        if normalized:
            prob /= prob.sum()
        return prob

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
