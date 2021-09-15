import numpy as np
import os
import logging
from os.path import join, dirname, realpath

from tprmp.models.tp_rmp import TPRMP
from tprmp.utils.loading import load_demos, load_demos_2d

_path_file = dirname(realpath(__file__))
DATA_PATH = join(_path_file, '..', '..', 'data', 'tasks')


class Experiment(object):
    logger = logging.getLogger(__name__)

    def __init__(self, **kwargs):
        self.verbose = kwargs.get('verbose', False)
        self.task = kwargs.get('task', 'test')
        self.demo_names = kwargs.get('demo_names', ['I'])
        self.demo_type = kwargs.get('demo_type', '2D')
        self.tag = kwargs.get('tag', None)
        self.dt = kwargs.get('dt', 0.01)
        self.save_model = kwargs.get('save_model', True)
        self.experiment_path = kwargs.get('experiment_path', join(DATA_PATH, self.task, 'experiments'))
        self.demo_path = kwargs.get('demo_path', join(DATA_PATH, self.task, 'demos'))
        os.makedirs(self.data_path, exist_ok=True)
        # training params
        self.test_comps = kwargs.get('test_comps', [3, 5, 7, 9])
        self.displace_var = kwargs.get('displace_var', 0.01 * np.arange(10))
        self.stiff_scale = kwargs.get('stiff_scale', 1.)
        self.mass_scale = kwargs.get('mass_scale', 1.)
        self.var_scale = kwargs.get('var_scale', 1.)
        self.delta = kwargs.get('delta', 2.)
        self.models = []
        self.demos = []
        # records data
        self.tracking_error = []
        self.goal_error = []
        self.success_rate = []
        self.rmpflow_goal_error = []
        self.rmpflow_success_rate = []

    def load_demos(self):
        for i, n in enumerate(self.demo_names):
            data_file = join(self.demo_path, n + '.p')
            if self.demo_type == '2D':
                self.demos[i] = load_demos_2d(data_file, dt=self.dt)
            elif self.demo_type == '6D':
                self.demos[i] = load_demos(data_file, tag=self.tag)

    def train(self):
        for i, n in enumerate(self.demo_names):
            for num_comp in self.test_comps:
                model = TPRMP(num_comp=num_comp, name=self.task, stiff_scale=self.stiff_scale, mass_scale=self.mass_scale, var_scale=self.var_scale, delta=self.delta)
                model.train(self.demos[i])
                model.save(name=n + '_' + str(num_comp) + '.p')

    def tracking_experiment(self):
        pass

    def adaptation_experiment(self, disturb=False, moving=False):
        pass

    def composable_experiment(self, moving=False):
        pass
