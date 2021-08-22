import os
from os.path import join, exists
import logging
import numpy as np
import pickle
import time

from tprmp.models.tp_hsmm import TPHSMM
from tprmp.models.rmp import compute_policy, compute_riemannian_metric, compute_potentials, compute_obsrv_prob
from tprmp.models.coriolis import compute_coriolis_force
from tprmp.optimizer.dynamics import optimize_dynamics
from tprmp.utils.loading import load


_path_file = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(_path_file, '..', '..', 'data', 'tasks')


class TPRMP(object):
    '''
    Wrapper of TPHSMM to retrieve RMP.
    '''
    logger = logging.getLogger(__name__)

    def __init__(self, **kwargs):
        self._sigma = kwargs.pop('sigma', 1.)
        self._stiff_scale = kwargs.pop('stiff_scale', 1.)
        self._tau = kwargs.pop('tau', 1.)
        self._potential_method = kwargs.pop('potential_method', 'quadratic')
        self._d_scale = kwargs.pop('d_scale', 1.)
        self._model = TPHSMM(**kwargs)
        self._global_mvns = None
        self._phi0 = None
        self._d0 = None
        self._R_net = None

    def save(self, name=None):
        self.model.save(name)
        file = join(DATA_PATH, self.model.name, 'models', name if name is not None else ('dynamics_' + str(time.time()) + '.p'))
        os.makedirs(file, exist_ok=True)
        with open(file, 'wb') as f:
            pickle.dump({'phi0': self._phi0, 'd0': self._d0}, f)

    def generate_global_gmm(self, frames):
        self._global_mvns = self.model.generate_global_gmm(frames)

    def retrieve(self, x, dx, frames, compute_global_mvns=False):
        """
        Retrieve global RMP canonical form.
        """
        M, f = self.rmp(x, dx, frames, compute_global_mvns=compute_global_mvns)
        return np.linalg.inv(M) @ f

    def rmp(self, x, dx, frames, compute_global_mvns=False):
        """
        Retrieve global RMP.
        """
        if compute_global_mvns or self._global_mvns is None:
            self.generate_global_gmm(frames)
        f = self.compute_global_policy(x, dx, frames) - compute_coriolis_force(x, dx, self._global_mvns)
        M = compute_riemannian_metric(x, self._global_mvns)
        return M, f

    def compute_global_policy(self, x, dx, frames):
        if not set(self.model.frame_names).issubset(set(frames)):
            raise IndexError(f'[TPRMP]: Frames must be subset of {self.model.frame_names}')
        policies = np.zeros((len(frames), self.model.manifold.dim_T))
        weights = self.compute_frame_weights(x, frames)
        for i, f_key in enumerate(self.model.frame_names):
            # compute local policy
            lx = frames[f_key].pullback(x)
            ldx = frames[f_key].pullback_tangent(dx)
            local_policy = compute_policy(self._phi0[f_key], self._d_scale * self._d0[f_key], lx, ldx, self.model.get_local_gmm(f_key),
                                          stiff_scale=self._stiff_scale, tau=self._tau, potential_method=self._potential_method)
            policies[i] = weights[f_key] * frames[f_key].transform_tangent(local_policy)
        return policies.sum(0)

    def compute_frame_weights(self, x, frames, normalized=True, eps=1e-30):
        origin = self.model.manifold.get_origin()
        frame_origins = {k: v.transform(origin) for k, v in frames.items()}
        weights = {}
        for f, o in frame_origins.items():
            v = self.model.manifold.log_map(x, base=o)
            w = np.exp(-v.T @ v / (2 * self._sigma ** 2))
            weights[f] = w
        s = sum(weights.values())
        if normalized:
            if s > eps:
                for f in weights:
                    weights[f] /= s
        return weights

    def compute_potential_field(self, x, frames):
        frame_weights = self.compute_frame_weights(x, frames)
        Phi = 0.
        for f_key in frames:
            lx = frames[f_key].pullback(x)
            mvns = self.model.get_local_gmm(f_key)
            weights = compute_obsrv_prob(lx, mvns)
            phi = compute_potentials(self.phi0[f_key], lx, mvns, stiff_scale=self._stiff_scale, tau=self._tau, potential_method=self._potential_method)
            Phi += frame_weights[f_key] * (weights.T @ phi)
        return Phi

    def compute_potential_field_frame(self, lx, frame):
        mvns = self.model.get_local_gmm(frame)
        weights = compute_obsrv_prob(lx, mvns)
        phi = compute_potentials(self.phi0[frame], lx, mvns, stiff_scale=self._stiff_scale, tau=self._tau, potential_method=self._potential_method)
        Phi = weights.T @ phi
        return Phi

    def train(self, demos, **kwargs):
        """
        Trains the TP-RMP with a given set of demonstrations.

        Parameters
        ----------
        :param demos: list of Demonstration objects
        """
        optimize_method = kwargs.get('optimize_method', 'flow')
        alpha = kwargs.get('alpha', 1e-5)
        beta = kwargs.get('beta', 1e-5)
        min_d = kwargs.get('min_d', 20.)
        energy = kwargs.get('energy', 0.)
        var_scale = kwargs.get('var_scale', 1.)
        verbose = kwargs.get('verbose', False)
        # train TP-HSMM/TP-GMM
        self.model.train(demos, **kwargs)
        if 'S' in self.model.manifold.name:  # decouple orientation and position
            pos_idx, quat_idx = self.model.manifold.get_pos_quat_indices(tangent=True)
            self.model.reset_covariance(pos_idx, quat_idx)
        if var_scale > 1.:
            self.model.scale_covariance(var_scale)
        # train dynamics
        self._phi0, self._d0 = optimize_dynamics(self.model, demos, alpha=alpha, beta=beta, optimize_method=optimize_method,
                                                 stiff_scale=self._stiff_scale, tau=self._tau, potential_method=self._potential_method, min_d=min_d, energy=energy, verbose=verbose)
        # train local Riemannian metrics TODO: RiemannianNetwork is still under consideration
        # self._R_net = optimize_riemannian_metric(self, demos, **kwargs)

    @staticmethod
    def load(task_name, model_name='sample.p'):
        """
        Parameters
        ----------
        :param model_name: name of model in data/models
        """
        tprmp = TPRMP()
        tprmp._model = TPHSMM.load(task_name, 'stats_' + model_name)
        file = join(DATA_PATH, task_name, 'models', 'dynamics_' + model_name)
        if not exists(file):
            raise ValueError(f'[TPHSMM]: File {file} does not exist!')
        dynamics = load(file)
        tprmp._phi0, tprmp._d0 = dynamics['phi0'], dynamics['d0']
        return tprmp

    @property
    def name(self):
        return self._model.name

    @property
    def model(self):
        return self._model

    @property
    def phi0(self):
        return self._phi0

    @property
    def d0(self):
        return self._d0

    @property
    def task_parameters(self):
        return self._model.frame_names

    @property
    def dt(self):
        return self._model.dt
