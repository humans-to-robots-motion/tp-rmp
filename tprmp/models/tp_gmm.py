import numpy as np
import time
import pickle
import os
from os.path import join, dirname, realpath, exists

from tprmp.utils.loading import load

_path_file = dirname(realpath(__file__))
DATA_PATH = join(_path_file, '..', '..', 'data', 'tasks')


class TPGMM(object):
    """
    "A Tutorial on Task-Parameterized Movement Learning and Retrieval." Sylvain Calinon, 2016.
    """

    def __init__(self, num_comp=5, name=''):
        """
        Parameters
        ----------
        :param num_comp: the number of components in the mixture
        """
        self._name = name
        if not isinstance(num_comp, int) or num_comp < 1:
            raise ValueError('[TPGMM]: num_comp must be a positive integer.')
        self._num_comp = num_comp
        self._mvns = None
        self._pi = None
        self._frame_names = None
        self._component_names = None
        self._num_frames = None
        self._dim_M = None
        self._tag_to_comp_map = None

    def train(self, demos):
        # TODO: implement training routine for TPGMM if needed
        pass

    def set_params(self, model_params):
        """
        Parameters
        ----------
        :param model_params: dictionary containing "mvns" and "pi"
            model_params["mvns"] = list of dict, each mvns[k] has the keys specified in self._frame_names and contains
                the Gaussian distribution of component k in the corresponding frame.
            model_params["pi"] = np.array([w1, w2, ... wK]).
            model_params["dim_M"] int, the dimension of the data.
        """
        self._pi = model_params['pi']
        self._mvns = model_params['mvns']
        self._num_comp = len(self._mvns)
        self._component_names = []
        for comp in range(self.num_comp):
            self._component_names.append(self.name + ("_comp_%s" % comp))
        self._frame_names = list(self.mvns[0].keys())
        self._num_frames = len(self._frame_names)
        self._dim_M = model_params['dim_M']
        self._tag_to_comp_map = model_params['tag_to_comp_map'] if 'tag_to_comp_map' in model_params else None

    def reset_covariance(self, idxs_1, idxs_2):
        """
        Reset the covariance matrices of all components at all frames to 0.
        Decouple params corresponded to idxs_1, idxs_2.
        """
        for k in range(self.num_comp):
            for f_key in self.frame_names:
                self._mvns[k][f_key].cov[np.ix_(idxs_1, idxs_2)] = 0.0
                self._mvns[k][f_key].cov[np.ix_(idxs_2, idxs_1)] = 0.0
                self._mvns[k][f_key].recompute_inv()
    
    def scale_covariance(self, scale):
        for k in range(self.num_comp):
            for f_key in self.frame_names:
                self._mvns[k][f_key].cov = self._mvns[k][f_key].cov * scale**2

    def get_local_gmm(self, frame, tag=None):
        if frame not in self.frame_names:
            raise ValueError(f'[TPGMM]: Frame {frame} not in model!')
        comps = range(self.num_comp) if ((self._tag_to_comp_map is None) or
                                         (tag not in self._tag_to_comp_map)) else self.tag_to_comp_map[tag]
        local_gmm = []
        for k in comps:
            local_gmm.append(self.mvns[k][frame])
        return local_gmm

    def generate_global_gmm(self, frames, tag=None):
        comps = range(self.num_comp) if ((self._tag_to_comp_map is None) or
                                         (tag not in self._tag_to_comp_map)) else self.tag_to_comp_map[tag]
        global_gmm = []
        for k in comps:
            global_gmm.append(self.combine_gaussians(k, frames))
        return global_gmm

    def combine_gaussians(self, k, frames):
        if not set(self.frame_names).issubset(set(frames)):
            raise IndexError(f'[TPGMM]: Frames must be subset of {self.frame_names}')
        g_list = []
        for f_key in self.frame_names:
            g_list.append(self.mvns[k][f_key].transform(frames[f_key].A, frames[f_key].b))  # Transform Gaussians to global frame
        return self.manifold.gaussian_product(g_list)

    def rename_component(self, comp, name):
        self.component_names[comp] = name

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def parameters(self):
        params = {
            'mvns': self.mvns,
            'pi': self.pi,
            'dim_M': self.dim_M,
            'tag_to_comp_map': self.tag_to_comp_map
        }
        return params

    def save(self):
        file = join(DATA_PATH, self.name, 'models', 'tpgmm_' + str(time.time()) + '.p')
        os.makedirs(file, exist_ok=True)
        with open(file, 'wb') as f:
            pickle.dump(self.parameters(), f)

    @staticmethod
    def load(task_name, model_name):
        file = join(DATA_PATH, task_name, 'models', model_name)
        if not exists(file):
            raise ValueError(f'[TPGMM]: File {file} does not exist!')
        model_params = load(file)
        model = TPGMM(name=task_name)
        model.set_params(model_params)
        return model

    @property
    def mvns(self):
        return self._mvns

    @property
    def manifold(self):
        return self._mvns[0][self._frame_names[0]].manifold

    @property
    def pi(self):
        return self._pi

    @property
    def num_comp(self):
        return self._num_comp

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def dim_M(self):
        return self._dim_M

    @property
    def name(self):
        return self._name

    @property
    def frame_names(self):
        return self._frame_names

    @property
    def component_names(self):
        return self._component_names

    @property
    def tag_to_comp_map(self):
        return self._tag_to_comp_map
