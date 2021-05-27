import os
import logging
import numpy as np

import torch
from torch import nn
from torch import optim
from tqdm import tqdm  # Displays a progress bar

from tprmp.networks.rmp_net import DeepRMPNetwork, device


_path_file = os.path.dirname(os.path.realpath(__file__))


class TPRMP(object):
    '''
    This current working with Deep RMP Network. TODO: extend to admit other model if neccesary
    '''
    logger = logging.getLogger(__name__)

    def __init__(self, name='test', dt=0.1):
        self.task_path = os.path.join(_path_file, '..', '..', 'data', 'tasks', name)
        os.makedirs(self.task_path, exist_ok=True)
        self._name = name
        self._task_parameters = None
        self._rmps = None
        self._dt = dt

    def load_models(self):
        """
        Parameters
        ----------
        :param model_name: name of model in data/models
        """
        self._rmps = {}
        model_path = os.path.join(self.task_path, 'models')
        os.makedirs(model_path, exist_ok=True)
        for file in os.listdir(model_path):
            if file.endswith('.pt') or file.endswith('.pth'):
                self._rmps[file.split('.')[0]] = torch.load(file)

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
        self._task_parameters = demos[0].frame_names
        self.rmps = {}
        for frame in self._task_parameters:
            self.rmps[frame] = self.train_network(frame, demos, **kwargs)

    def train_network(self, frame, demos, **kwargs):
        """
        Trains a DeepRMPNet for each frame. TODO: develop treatment for quaternion!

        Parameters
        ----------
        :param demos: list of Demonstration objects
        """
        if frame not in demos[0].frame_names:
            TPRMP.logger.info('Frame %s does not exist in the demonstrations!' % frame)
            return
        hidden_dim = kwargs.get('hidden_dim', 64)
        num_epoch = kwargs.get('num_epoch', 100)
        lr = kwargs.get('lr', 5e-3)
        lr_step_size = kwargs.get('lr_step_size', 40)
        lr_gamma = kwargs.get('lr_gamma', 0.5)
        weight_decay = kwargs.get('weight_decay', 1e-3)
        grad_norm_bound = kwargs.get('grad_norm_bound', 10.0)
        dim_M = demos[0].dim_M  # treatment for quaternion?
        # setting up
        model = DeepRMPNetwork(dim_M, hidden_dim=hidden_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
        # training routine
        TPRMP.logger.info('Start training RMPNet for frame %s' % frame)
        model.train()  # Set the model to training mode
        for i in tqdm(range(num_epoch)):
            running_loss = []
            for demo in demos:
                traj, d_traj, dd_traj = demo.traj_in_frames[frame].values()
                for t in range(demo.length):
                    x, dx, ddx = torch.from_numpy(traj[:, t]).to(device), torch.from_numpy(d_traj[:, t]).to(device), torch.from_numpy(dd_traj[:, t]).to(device)
                    optimizer.zero_grad()
                    M, c, g = model(x, dx)
                    rmp = M @ ddx + c + g  # output current RMP
                    loss = criterion(rmp, 0)  # enforce conservative structure
                    running_loss.append(loss.item())
                    loss.backward()  # Backprop gradients to all tensors in the network
                    torch.nn.utils.clip_grad_norm(model.parameters(), grad_norm_bound)
                    optimizer.step()  # Update trainable weights
            scheduler.step()
            if i % 5 == 0:
                TPRMP.logger.info("Epoch {} loss:{}".format(i + 1, np.mean(running_loss)))

    @property
    def rmps(self):
        return self._rmps

    @property
    def name(self):
        return self._name

    @property
    def sampling_time(self):
        return self._dt
