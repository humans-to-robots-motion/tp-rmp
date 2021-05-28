import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from tqdm import tqdm  # Displays a progress bar
from os.path import join, dirname, abspath

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.load_demos import load_demos  # noqa
from tprmp.networks.rmp_net import DeepRMPNetwork, device  # noqa
from tprmp.demonstrations.trajectory import compute_traj_derivatives  # noqa

dt = 0.1
hidden_dim = 64
num_epoch = 5
lr = 5e-3
lr_step_size = 40
lr_gamma = 0.5
weight_decay = 1e-3
grad_norm_bound = 10.0


def train(model, demos):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    # training routine
    print('Start training')
    model.train()  # Set the model to training mode
    for i in tqdm(range(num_epoch)):
        running_loss = []
        for demo in demos:
            traj, d_traj, dd_traj = demo.values()
            for t in range(traj.shape[1]):
                x, dx, ddx = torch.from_numpy(traj[:, t]).unsqueeze(0).to(device), torch.from_numpy(d_traj[:, t]).unsqueeze(0).to(device), torch.from_numpy(dd_traj[:, t]).to(device)
                optimizer.zero_grad()
                M, c, g = model(x, dx)
                rmp = M @ ddx + c + g  # output current RMP
                loss = criterion(rmp, torch.zeros(2, device=device))  # enforce conservative structure
                running_loss.append(loss.item())
                loss.backward()  # Backprop gradients to all tensors in the network
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_bound)
                optimizer.step()  # Update trainable weights
        scheduler.step()
        if i % 10 == 0:
            print("Epoch {} loss:{}".format(i + 1, np.mean(running_loss)))


def retrieve(model, x0, dx0, T):
    X, Y = [x0[0]], [x0[1]]
    prev_x = x0.astype(np.float32)
    prev_dx = dx0.astype(np.float32)
    for t in np.linspace(0, T, int(T / dt) + 1):
        x, dx = torch.from_numpy(prev_x).unsqueeze(0).to(device), torch.from_numpy(prev_dx).unsqueeze(0).to(device)
        M, c, g = model(x, dx)
        M, c, g = M.cpu().detach().numpy(), c.cpu().detach().numpy(), g.cpu().detach().numpy()
        ddx = -np.linalg.inv(M) @ (c + g)
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
    M, c, g = model(p, v)
    g = g.cpu().detach().numpy()
    plt.quiver(x, y, g[:, 0], g[:, 1])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Example run: python train_deeprmp.py test.p')
    parser.add_argument('task', help='The task folder', type=str, default='test')
    parser.add_argument('data', help='The data file', type=str, default='data.p')
    args = parser.parse_args()

    DATA_DIR = join(ROOT_DIR, 'data', 'tasks', args.task, 'demos')
    data_file = join(DATA_DIR, args.data)
    trajs = load_demos(data_file)
    # preprocess data
    dim_M = len(trajs[0])
    demos = []
    for traj in trajs:
        X = np.array(list(zip(traj[0], traj[1]))).T
        X, dX, ddX = compute_traj_derivatives(X, dt)
        demos.append({'traj': X.astype(np.float32), 'd_traj': dX.astype(np.float32), 'dd_traj': ddX.astype(np.float32)})

    print('Viewing demos, close to train...')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Demos')
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    for traj in trajs:
        ax.plot(traj[0], traj[1], marker="o", color="r", markersize=2)
    plt.show()
    model = DeepRMPNetwork(dim_M, hidden_dim=hidden_dim).to(device)
    train(model, demos)
    draw_force_pattern(model)
    retrieve(model, np.array([0., 4.]), np.zeros(2), 3)
