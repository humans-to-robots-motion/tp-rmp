import os
import logging

from tprmp.models.tp_hsmm import TPHSMM


_path_file = os.path.dirname(os.path.realpath(__file__))


class TPRMP(object):
    '''
    Wrapper of TPHSMM to retrieve RMP.
    '''
    logger = logging.getLogger(__name__)

    def __init__(self, **kwargs):
        name = kwargs.get('name', '')
        self.task_path = os.path.join(_path_file, '..', '..', 'data', 'tasks', name)
        os.makedirs(self.task_path, exist_ok=True)
        self._model = TPHSMM(**kwargs)

    def save(self, name=None):
        self._model.save(name)

    def load(self, model_name='sample'):
        """
        Parameters
        ----------
        :param model_name: name of model in data/models
        """
        self._model = TPHSMM.load(self._name, model_name + '.p')

    def retrieve(self):
        """
        Retrieve global RMP.
        """
        if not self._rmps:
            return

    def train(self, demos, **kwargs):
        """
        Trains the TP-RMP with a given set of demonstrations.

        Parameters
        ----------
        :param demos: list of Demonstration objects
        """
        self._model.train(demos, **kwargs)

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
