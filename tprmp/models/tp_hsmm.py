from tprmp.demonstrations.base import Demonstration
import numpy as np
import time
import pickle
import os
import logging
from sys import float_info
import matplotlib.pyplot as plt  # noqa
from os.path import join, dirname, realpath, exists

from tprmp.models.tp_gmm import TPGMM
from tprmp.optimizer.em import EM
from tprmp.utils.loading import load
from tprmp.visualization.demonstration import plot_demo
from tprmp.visualization.models import plot_hsmm, plot_gmm
from tprmp.visualization.em import plot_gamma

_path_file = dirname(realpath(__file__))
DATA_PATH = join(_path_file, '..', '..', 'data', 'tasks')


class TPHSMM(TPGMM):
    """
    "A Tutorial on Task-Parameterized Movement Learning and Retrieval." Sylvain Calinon, 2016.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, num_comp=0, name=''):
        super(TPHSMM, self).__init__(num_comp, name)
        self._trans_prob = None
        self._duration_prob = None
        self._max_duration = None
        self._dt = None
        self._end_states = None

    def train(self, demos, **kwargs):
        """
        Trains the TP-HSMM with a given set of demonstrations.
        """
        with_tag = kwargs.get('with_tag', False)
        num_comp_per_tag = kwargs.get('num_comp_per_tag', None)
        hmm_shape = kwargs.get('hmm_shape', 'full')
        if isinstance(demos, Demonstration):
            demos = [demos]
        if (hmm_shape != 'full'):
            if hmm_shape == 'straight-line':
                topology = (np.eye(self.num_comp) + np.diag(np.ones(self.num_comp - 1), 1))
            elif hmm_shape.startswith('split-'):  # split structure for multi sub-behavior skills
                split_n = int(hmm_shape[6:])
                TPHSMM.logger.info(f'Split-{split_n} structure is used!')
                if with_tag:
                    demo_tags = list(set([demo.tag for demo in demos]))
                    if demo_tags == [None]:
                        raise ValueError('[TPHSMM]: Demos are not tagged for trajectory cluster')
                    if (not num_comp_per_tag):
                        num_comp_per_tag = TPHSMM.split_equal_comp_per_tag(self.num_comp, demo_tags)
                if num_comp_per_tag:  # overwrite
                    nb_comp_per_line = tuple(num_comp_per_tag)
                else:
                    nb_comp_per_line = TPHSMM.split_equal_comp_per_tag(self.num_comp, range(split_n))
                # construct split topology
                off_diag = np.ones(self.num_comp - 1)
                comp_n = 0
                for _, nb_comp in nb_comp_per_line[0:-2]:
                    comp_n += nb_comp
                    off_diag[comp_n - 1] = 0.0
                topology = (np.eye(self.num_comp) + np.diag(off_diag, 1))
                # remove connection between splits
                k = nb_comp_per_line[0][1]
                for i in range(len(nb_comp_per_line) - 1):
                    topology[k - 1, k] = 0.
                    k += nb_comp_per_line[i][1]
            else:
                TPHSMM.logger.warn('HSMM shape is not recognized. Use full structure')
                topology = np.ones((self.num_comp, self.num_comp))
        else:
            topology = np.ones((self.num_comp, self.num_comp))
        # start training
        em = EM(demos, num_comp=self.num_comp, topology=topology, **kwargs)
        em.optimize()
        self.set_params(em.model_parameters)

    def plot_model(self, demos, plot_quat=False, show=True):
        if isinstance(demos, Demonstration):
            demos = [demos]
        # fig = plt.figure(f'TPHSMM of {self.name}', figsize=(18, 6))
        # ax_g = fig.add_subplot(1, 3, 1)
        # ax_d = fig.add_subplot(1, 3, 2, projection='3d')
        # ax_h = fig.add_subplot(1, 3, 3)
        plot_hsmm(self, new_fig=True, show=show)
        # plt.sca(ax_g)
        plot_gamma(self.gamma, new_fig=True, show=show)
        # plt.sca(ax_d)
        tag_to_demos = {}
        for i, demo in enumerate(demos):
            if demo.tag not in tag_to_demos:
                tag_to_demos[demo.tag] = []
            tag_to_demos[demo.tag].append(i)
        for tag in tag_to_demos:  # plot one demo for each tag
            m = tag_to_demos[tag][0]
            plot_demo(demos[m], plot_quat=plot_quat, new_fig=True, show=False)
            plot_gmm(self, demos[m].get_task_parameters(), plot_quat=plot_quat, tag=tag, new_fig=False, show=show)
        # if show:
        #     plt.show()

    def set_params(self, model_params):
        super(TPHSMM, self).set_params(model_params)
        self._trans_prob = model_params["trans_prob"]
        self._duration_prob = model_params["duration_prob"]
        self._max_duration = int(model_params["max_duration"])
        self._end_states = model_params["end_states"]
        self._dt = model_params["dt"]
        self._gamma = model_params["gamma"]

    def parameters(self):
        params = super(TPHSMM, self).parameters()
        params.update({
            'trans_prob': self.trans_prob,
            'duration_prob': self.duration_prob,
            'max_duration': self.max_duration,
            'end_states': self.end_states,
            'gamma': self.gamma,
            'dt': self.dt
        })
        return params

    def get_end_components(self, frames, num_comp=1):
        """
        Get the end components of TP-HSMM given current frames.

        Parameters
        ----------
        :param frames: dict, frames[frame_name] is a Frame object and contains the task
               parameters frame_name.
        :param num_comp: int. Number of components to be abstracted as end components.

        Returns
        -------
        :return: end_gaus_all: dict, {component_idx:gaus} where component_idx is the index and
                 gaus is associated Gaussian in global frame.
        """
        end_components = np.argpartition(self.end_states, -num_comp)[-num_comp:]
        end_gaus_all = dict()
        for end_comp in end_components:
            end_gaus = self.combine_gaussians(end_comp, frames)
            end_gaus_all[end_comp] = end_gaus
        return end_gaus_all

    def compute_pdfs(self, frames, obsrv):
        """
        Compute the pdfs over all components of the TPHSMM given the current frames and the observation.

        Parameters
        ----------
        :param frames: dict, frames[frame_name] is a Frame object and contains the task
               parameters frame_name.
        :param obsrv: np.array of the appropriate dim.

        Returns
        -------
        :return: pdfs (list): list of log pdf over the global gauss of all components in the TPHSMM.
        """
        pdfs = np.zeros(self.num_comp)
        for k in range(self.num_comp):
            pdfs[k] = self.combine_gaussians(k, frames).pdf(obsrv) + float_info.min
        return pdfs

    def save(self, name=None):
        model_folder = join(DATA_PATH, self.name, 'models')
        os.makedirs(model_folder, exist_ok=True)
        file = join(model_folder, (name if name is not None else 'tphsmm_' + str(time.time())) + '.p')
        with open(file, 'wb') as f:
            pickle.dump(self.parameters(), f)

    @staticmethod
    def split_equal_comp_per_tag(num_comp, tags):
        num_comp_per_tag = []
        eq_nb = num_comp // len(tags)
        for k, tag in enumerate(tags):
            part_nb = eq_nb
            if k < num_comp % len(tags):
                part_nb = eq_nb + 1
            num_comp_per_tag.append((tag, part_nb))
        return tuple(num_comp_per_tag)

    @staticmethod
    def load(task_name, model_name):
        file = join(DATA_PATH, task_name, 'models', model_name)
        if not exists(file):
            raise ValueError(f'[TPHSMM]: File {file} does not exist!')
        model_params = load(file)
        model = TPHSMM(name=task_name)
        model.set_params(model_params)
        return model

    @property
    def trans_prob(self):
        return self._trans_prob

    @property
    def duration_prob(self):
        return self._duration_prob

    @property
    def max_duration(self):
        return self._max_duration

    @property
    def dt(self):
        return self._dt

    @property
    def end_states(self):
        return self._end_states

    @property
    def gamma(self):
        return self._gamma
