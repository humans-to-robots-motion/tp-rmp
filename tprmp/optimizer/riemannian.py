import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm  # Displays a progress bar
from collections import deque

from tprmp.demonstrations.quaternion import q_to_euler
from tprmp.models.rmp import compute_potential_term, compute_obsrv_prob
from tprmp.networks.riemannian_net import RiemannianNetwork

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_riemannian_net(tp_gmm, demos, phi0, **kwargs):  # NOTE: this currently only works with R^n or R^3 x S^3
    model = RiemannianNetwork(tp_gmm.manifold.dim_T, **kwargs).to(device)
    # training hyperparams
    lr = kwargs.get('lr', 0.005)
    weight_decay = kwargs.get('weight_decay', 3e-5)
    # grad_norm_bound = kwargs.get('grad_norm_bound', 1.)
    max_epoch = kwargs.get('max_epoch', 25)
    batch_size = kwargs.get('batch_size', 100)
    # reg_lambda = kwargs.get('reg_lambda', 0.05)
    log_window = kwargs.get('log_window', 100)
    # potential params
    stiff_scale = kwargs.get('stiff_scale', 1.)
    tau = kwargs.get('tau', 0.05)
    potential_method = kwargs.get('potential_method', 'quadratic')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    running_loss = deque(maxlen=log_window)
    loader = create_dataloader(tp_gmm, demos, phi0, batch_size=batch_size, stiff_scale=stiff_scale, tau=tau, potential_method=potential_method)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma) NOTE: may not need scheduler
    # training routine
    logger.info('Training local Riemannian metric...')
    model.train()  # Set the model to training mode
    i = 0
    for epoch in tqdm(range(max_epoch)):
        for bid, (x, dx, pf) in enumerate(loader):
            x, dx, pf = x.to(device), dx.to(device).detach(), pf.to(device).detach()
            optimizer.zero_grad()
            _, M_inv, _, _ = model(x, dx)
            warped_term = torch.matmul(M_inv, pf).squeeze()
            warped_term_n = torch.linalg.norm(warped_term, dim=1, keepdim=True).detach()
            warped_term_normalized = warped_term.div(warped_term_n)
            dxn = torch.linalg.norm(dx, dim=1, keepdim=True).detach()
            dx_normalized = dx.div(dxn)
            loss = criterion(warped_term_normalized, dx_normalized)  # + reg_lambda * torch.linalg.norm(M)
            running_loss.append(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_bound)
            optimizer.step()
            # scheduler.step()
            i += 1
            if i % log_window == 0:
                logger.debug(f'Epoch {epoch} loss:{np.mean(running_loss)}')
    logger.info(f'Final loss: {loss.item()}')


def create_dataloader(tp_gmm, demos, phi0, batch_size=5, **kwargs):
    dim_M, dim_T = demos[0].manifold.dim_M, demos[0].manifold.dim_T
    xs, dxs, pfs = np.zeros((dim_M, 0), dtype=np.float32), np.zeros((dim_T, 0), dtype=np.float32), np.zeros((0, dim_T), dtype=np.float32)
    for demo in demos:
        mvns = tp_gmm.generate_global_gmm(demo.get_task_parameters())
        x = demo.traj.astype(np.float32)
        if 'S^3' in demos[0].manifold.name:
            x = np.append(x[:3], q_to_euler((x[3:])), axis=0)
        xs = np.append(xs, x, axis=1)
        dxs = np.append(dxs, demo.d_traj.astype(np.float32), axis=1)
        pf = []
        weights = compute_obsrv_prob(x, mvns)
        for t in range(demo.length):
            pf.append(compute_potential_term(weights[:, t], phi0, x[:, t], mvns, **kwargs))
        pf = np.array(pf, dtype=np.float32)
        pfs = np.append(pfs, pf, axis=0)
    dataset = TensorDataset(torch.Tensor(xs.T), torch.Tensor(dxs.T), torch.Tensor(pfs))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader
