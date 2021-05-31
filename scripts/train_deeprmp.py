import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from tqdm import tqdm  # Displays a progress bar
from os.path import join, dirname, abspath
from collections import deque

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.load_demos import load_demos  # noqa
from tprmp.networks.rmp_net import DeepRMPNetwork  # noqa
from tprmp.demonstrations.trajectory import compute_traj_derivatives  # noqa
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 3
dt = 0.005
hidden_dim = 128
num_epoch = 20
lr = 5e-3
lr_step_size = 40
lr_gamma = 0.5
weight_decay = 1e-3
grad_norm_bound = 10.0
log_window = 10


def simple_demo(T, dt):
    nb = int(1 / dt * T)
    a = np.ones((2, nb))
    a[:, int(nb / 2):] = -1
    v = np.cumsum(a, axis=1) * dt
    X = np.cumsum(v, axis=1) * dt
    return [(X[0], X[1])]


def train(model, demos):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    # training routine
    print('Start training')
    model.train()  # Set the model to training mode
    for i in tqdm(range(num_epoch)):
        running_loss = deque(maxlen=log_window)
        for demo in demos:
            traj, d_traj, dd_traj = demo.values()
            for t in range(traj.shape[1]):
                x, dx, ddx = torch.from_numpy(traj[:, t]).unsqueeze(0).to(device), torch.from_numpy(d_traj[:, t]).unsqueeze(0).to(device), torch.from_numpy(dd_traj[:, t]).to(device)
                optimizer.zero_grad()
                M, c, g, B = model(x, dx)
                rmp = M @ ddx + B @ dx.squeeze(0) + c + g  # output current RMP
                loss = criterion(rmp, torch.zeros_like(rmp, device=device))  # enforce conservative structure
                running_loss.append(loss.item())
                loss.backward()  # Backprop gradients to all tensors in the network
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_bound)
                optimizer.step()  # Update trainable weights
        scheduler.step()
        if i % log_window == 0:
            print("Epoch {} loss:{}".format(i + 1, np.mean(running_loss)))


def retrieve(model, x0, dx0, T):
    X, Y = [x0[0]], [x0[1]]
    prev_x = x0.astype(np.float32)
    prev_dx = dx0.astype(np.float32)
    for t in np.linspace(0, T, int(T / dt) + 1):
        x, dx = torch.from_numpy(prev_x).unsqueeze(0).to(device), torch.from_numpy(prev_dx).unsqueeze(0).to(device)
        ddx = model.rmp(x, dx)
        ddx = ddx.cpu().detach().numpy()
        new_dx = prev_dx + ddx * dt
        new_x = prev_x + new_dx * dt
        X.append(new_x[0])
        Y.append(new_x[1])
        prev_dx, prev_x = new_dx, new_x
    plt.plot(X, Y, marker="o", color="b", markersize=5)
    plt.show()


def draw_force_pattern(model):
    x, y = np.meshgrid(np.linspace(0, 5, 10), np.linspace(0, 5, 10))
    p = np.vstack(map(np.ravel, (x, y))).T
    p = torch.from_numpy(p.astype(np.float32)).to(device)
    v = torch.zeros_like(p)
    M, c, g, B = model(p, v)
    g = g.cpu().detach().numpy()
    plt.quiver(x, y, g[:, 0], g[:, 1])
    plt.title('Potential g')
    plt.show()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    #                                  description='Example run: python train_deeprmp.py test.p')
    # parser.add_argument('task', help='The task folder', type=str, default='test')
    # parser.add_argument('data', help='The data file', type=str, default='test.p')
    # args = parser.parse_args()

    # DATA_DIR = join(ROOT_DIR, 'data', 'tasks', args.task, 'demos')
    # data_file = join(DATA_DIR, args.data)
    # trajs = load_demos(data_file)
    trajs = simple_demo(T, dt)
    # preprocess data
    dim_M = len(trajs[0])
    demos = []
    for traj in trajs:
        X = np.array(list(zip(traj[0], traj[1]))).T
        X, dX, ddX = compute_traj_derivatives(X, dt, smooth=True)
        demos.append({'traj': X.astype(np.float32), 'd_traj': dX.astype(np.float32), 'dd_traj': ddX.astype(np.float32)})
    print('Viewing demos, close to train...')
    for traj in trajs:
        plt.plot(traj[0], traj[1], marker="o", color="r", markersize=2)
    plt.show()
    model = DeepRMPNetwork(dim_M).to(device)
    train(model, demos)
    draw_force_pattern(model)
    retrieve(model, np.zeros(2), 0.1 * np.ones(2), T)
