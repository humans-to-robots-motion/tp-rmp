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

    def __init__(self, num_comp=5, name=''):
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
        plot_gamma_flag = kwargs.get('plot_gamma_flag', False)
        num_comp_per_tag = kwargs.get('num_comp_per_tag', None)
        if isinstance(demos, Demonstration):
            demos = [demos]
        # start training
        if with_tag:
            tags = list(set([demo.tag for demo in demos]))
            if num_comp_per_tag is None:
                num_comp_per_tag = TPHSMM.split_equal_comp_per_tag(self.num_comp, tags)
            self._tag_to_comp_map = {}
            n = 0
            for tag, nb_comp in num_comp_per_tag.items():
                self._tag_to_comp_map[tag] = np.arange(nb_comp) + n
                n += nb_comp
            TPHSMM.logger.info(f'Tagging: {self.tag_to_comp_map}')
            tag_to_demos = TPHSMM.compute_tag_to_demo(demos)
            tag_to_params = {}
            for tag in tags:
                em = EM(tag_to_demos[tag], num_comp=num_comp_per_tag[tag], **kwargs)
                em.optimize()
                tag_to_params[tag] = em.model_parameters
                if plot_gamma_flag:
                    plot_gamma(em.model_parameters['gamma'], title=f'Gamma of tag: {tag}', new_fig=True, show=True)
            self.set_params_with_tag(tag_to_params)
        else:
            em = EM(demos, num_comp=self.num_comp, **kwargs)
            em.optimize()
            self.set_params(em.model_parameters)
            if plot_gamma_flag:
                plot_gamma(em.model_parameters['gamma'], new_fig=True, show=True)

    def set_params_with_tag(self, tag_to_params):
        if self.tag_to_comp_map is None:
            return
        model_params = {}
        tags = list(tag_to_params.keys())
        model_params['mvns'] = np.array([{} for _ in range(self.num_comp)])
        model_params['pi'] = np.zeros(self.num_comp)
        model_params['trans_prob'] = np.zeros((self.num_comp, self.num_comp))
        model_params['end_states'] = np.zeros(self.num_comp)
        model_params['max_duration'] = tag_to_params[tags[0]]['max_duration']
        model_params['dim_M'] = tag_to_params[tags[0]]['dim_M']
        model_params['dt'] = tag_to_params[tags[0]]['dt']
        for tag in tag_to_params:
            model_params['mvns'][self.tag_to_comp_map[tag]] = tag_to_params[tag]['mvns']
            model_params['pi'][self.tag_to_comp_map[tag]] = tag_to_params[tag]['pi']
            model_params['trans_prob'][np.ix_(self.tag_to_comp_map[tag], self.tag_to_comp_map[tag])] = tag_to_params[tag]['trans_prob']
            model_params['max_duration'] = max(model_params['max_duration'], tag_to_params[tag]['max_duration'])
            model_params['end_states'][self.tag_to_comp_map[tag]] = tag_to_params[tag]['end_states']
            if model_params['dim_M'] != tag_to_params[tags[0]]['dim_M']:
                raise ValueError('[TPHSMM] dim_M is not consistent across tags')
            if model_params['dt'] != tag_to_params[tags[0]]['dt']:
                raise ValueError('[TPHSMM] dt is not consistent across tags')
        # duration prob
        model_params['duration_prob'] = np.zeros((self.num_comp, model_params['max_duration']))
        for tag in tag_to_params:
            model_params['duration_prob'][self.tag_to_comp_map[tag], :tag_to_params[tag]['max_duration']] = tag_to_params[tag]['duration_prob']
        # normalizing prob
        model_params['pi'] = model_params['pi'] / model_params['pi'].sum()
        model_params['end_states'] = model_params['end_states'] / model_params['end_states'].sum()
        self.set_params(model_params)

    def plot_model(self, demos, tagging=True, plot_quat=False, show=True):
        if isinstance(demos, Demonstration):
            demos = [demos]
        plot_hsmm(self, new_fig=True, show=show)
        tag_to_demos = TPHSMM.compute_tag_to_demo(demos)
        for tag in tag_to_demos:  # plot one demo for each tag
            demo = tag_to_demos[tag][0]
            plot_demo(demo, plot_quat=plot_quat, new_fig=True, show=False)
            plot_gmm(self, demo.get_task_parameters(), plot_quat=plot_quat, tag=tag if tagging else None, new_fig=False, show=show)

    def set_params(self, model_params):
        super(TPHSMM, self).set_params(model_params)
        self._trans_prob = model_params["trans_prob"]
        self._duration_prob = model_params["duration_prob"]
        self._max_duration = int(model_params["max_duration"])
        self._end_states = model_params["end_states"]
        self._dt = model_params["dt"]

    def parameters(self):
        params = super(TPHSMM, self).parameters()
        params.update({
            'trans_prob': self.trans_prob,
            'duration_prob': self.duration_prob,
            'max_duration': self.max_duration,
            'end_states': self.end_states,
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
    def compute_tag_to_demo(demos):
        tag_to_demos = {}
        for demo in demos:
            if demo.tag not in tag_to_demos:
                tag_to_demos[demo.tag] = []
            tag_to_demos[demo.tag].append(demo)
        return tag_to_demos

    @staticmethod
    def split_equal_comp_per_tag(num_comp, tags):
        num_comp_per_tag = {}
        eq_nb = num_comp // len(tags)
        for k, tag in enumerate(tags):
            part_nb = eq_nb
            if k < num_comp % len(tags):
                part_nb = eq_nb + 1
            num_comp_per_tag[tag] = part_nb
        return num_comp_per_tag

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
