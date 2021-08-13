import logging
from tprmp.models.tp_rmp import TPRMP
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


def optimize_riemannian_metric(tp_rmp, demos, **kwargs):  # NOTE: this currently only works with R^n or R^3 x S^3
    frames = demos[0].frame_names
    dim_T = demos[0].manifold.dim_T
    M_frames = {}
    for frame in frames:
        model = RiemannianNetwork(dim_T, **kwargs).to(device)
        train_riemannian_net(model, tp_rmp, demos, frame, **kwargs)
        model.eval()
        M_frames[frame] = model.cpu()
    return M_frames


def train_riemannian_net(model, tp_rmp, demos, frame, **kwargs):
    lr = kwargs.get('lr', 0.005)
    weight_decay = kwargs.get('weight_decay', 3e-5)
    # grad_norm_bound = kwargs.get('grad_norm_bound', 1.)
    max_epoch = kwargs.get('max_epoch', 25)
    batch_size = kwargs.get('batch_size', 100)
    # reg_lambda = kwargs.get('reg_lambda', 0.05)
    log_window = kwargs.get('log_window', 100)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    running_loss = deque(maxlen=log_window)
    loader = create_dataloader(demos, frame, batch_size=batch_size)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma) NOTE: may not need scheduler
    # training routine
    logger.info(f'Training local Riemannian metric of frame {frame}...')
    model.train()  # Set the model to training mode
    i = 0
    for epoch in tqdm(range(max_epoch)):
        for bid, (x, dx) in enumerate(loader):
            x_np = x.detach().cpu().numpy().T
            x, dx = x.to(device), dx.to(device).detach()
            optimizer.zero_grad()
            M, M_inv, _, _ = model(x, dx)
            mvns = tp_rmp.model.get_local_gmm(frame)
            weights = compute_obsrv_prob(x_np, mvns)
            potential_force = torch.Tensor(compute_potential_term(weights, tp_rmp.phi0[frame], x_np, mvns).T).unsqueeze(2).to(device)
            warped_term = torch.matmul(M_inv, potential_force).squeeze()
            warped_term_n = torch.linalg.norm(warped_term, dim=1, keepdim=True).detach()  # TODO: need test on 7DOFs robot
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


def create_dataloader(demos, frame, batch_size=5):
    dim_T = demos[0].manifold.dim_T
    xs, dxs = np.zeros((dim_T, 0), dtype=np.float32), np.zeros((dim_T, 0), dtype=np.float32)
    for demo in demos:
        trajs = demo.traj_in_frames[frame]
        x = trajs['traj'].astype(np.float32)
        if 'S^3' in demos[0].manifold.name:
            x = np.append(x[:3], q_to_euler((x[3:])), axis=0)
        xs = np.append(xs, x, axis=1)
        dxs = np.append(dxs, trajs['d_traj'].astype(np.float32), axis=1)
    dataset = TensorDataset(torch.Tensor(xs.T), torch.Tensor(dxs.T))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


if __name__ == '__main__':
    import numpy as np
    from tprmp.demonstrations.probability import ManifoldGaussian
    from tprmp.demonstrations.manifold import Manifold
    from tprmp.demonstrations.base import Demonstration
    from tprmp.models.tp_gmm import TPGMM
    from tprmp.visualization.dynamics import visualize_rmp
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    def simple_demo(T, dt):
        ddx = np.ones((2, T))
        ddx[:, int(T / 2):] = -1
        dx = np.cumsum(ddx, axis=1) * dt
        x = np.cumsum(dx, axis=1) * dt
        return x, dx, ddx

    manifold = Manifold.get_euclidean_manifold(2)
    num_comp = 10
    max_range = 4
    means = np.linspace(1, max_range, num_comp)
    var = (max_range - 1) / (2 * num_comp)
    scale_var = 1.
    mvns = [{'obj': ManifoldGaussian(manifold, means[k] * np.ones(2), scale_var * var * np.eye(2))} for k in range(num_comp)]
    T = 400
    dt = 0.01
    x, dx, ddx = simple_demo(T, dt)
    demo = Demonstration(x, dt=dt)
    demo.add_frame_from_pose(np.zeros(2), 'obj')
    tp_gmm = TPGMM(num_comp=num_comp)
    tp_gmm._mvns = mvns
    tp_gmm._frame_names = ['obj']
    phi0 = {}
    phi0['obj'] = [5.12158529e+1, 2.08098141e+1, 1.23384782e+1, 2.85985728, 1.04711665e-2, 1.04711353e-2, 2.00071447e-7, 1.86287912e-7, 1.79560689e-7, 1.78931385e-7]
    d0 = {}
    d0['obj'] = [5.83848224, 2.34618446e+1, 1.00000113e-2, 8.27816653, 1.00000581e-2, 9.76575512e-1, 5.72929509e-1, 9.95741283e-1, 1.00000060e-2, 7.68730219]
    tp_rmp = TPRMP()
    tp_rmp._model = tp_gmm
    tp_rmp._phi0 = phi0
    tp_rmp._d0 = d0
    # test training
    M_frames = optimize_riemannian_metric(tp_rmp, [demo])
    # test retrieval
    M_frames['obj'].eval()
    x0, dx0 = np.array([0, 0.5]), np.zeros(2)
    visualize_rmp(phi0['obj'], d0['obj'], tp_gmm.get_local_gmm('obj'), x0, dx0, T, dt, limit=10, R_net=M_frames['obj'])
