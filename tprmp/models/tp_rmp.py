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

    def save(self, name=None):
        self.model.save(name)

    def retrieve(self, frames, **kwargs):
        """
        Retrieve global RMP.
        """
        pass

    def compute_potential_force(self, x, t, global_mvns, **kwargs):
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

    def compute_force_weights(self, x, t, global_mvns, **kwargs):
        alpha_pose = kwargs.get('alpha_pose', 0.5)
        alpha_time = kwargs.get('alpha_time', 1.0)
        eps_pose = kwargs.get('eps_pose', 0.05)
        eps_time = kwargs.get('eps_time', 0.01)
        normalized = kwargs.get('normalized', True)
        duration_mvns = self.model.duration_mvns
        tag_to_comp_map = self.model.tag_to_comp_map
        weights = np.zeros(self.model.num_comp)
        for tag in tag_to_comp_map:
            tau = 0.
            for k in tag_to_comp_map[tag]:
                # pose weighting
                manifold = global_mvns[k].manifold
                pose, pose_cov = global_mvns[k].mean, global_mvns[k].cov + eps_pose * np.eye(manifold.dim_T)
                projected_pose = manifold.log_map(x, base=pose)
                weights[k] = np.squeeze(np.exp(-alpha_pose * projected_pose.T @ np.linalg.inv(pose_cov) @ projected_pose))
                # time weighting
                tau += duration_mvns[k].mean
                tau_cov = duration_mvns[k].cov + eps_time
                weights[k] *= np.squeeze(np.exp(-alpha_time * ((t - tau) ** 2) / tau_cov))
        # normalize weights
        if normalized:
            weights /= weights.sum()
        return weights

    def train(self, demos, **kwargs):
        """
        Trains the TP-RMP with a given set of demonstrations.

        Parameters
        ----------
        :param demos: list of Demonstration objects
        """
        self.model.train(demos, **kwargs)

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
