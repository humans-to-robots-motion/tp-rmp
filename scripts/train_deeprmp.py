import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from tqdm import tqdm  # Displays a progress bar
import os
from os.path import join, dirname, abspath
from collections import deque
import time

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load_demos  # noqa
from tprmp.networks.rmp_net import DeepRMPNetwork  # noqa
from tprmp.demonstrations.trajectory import compute_traj_velocity  # noqa
from tprmp.demonstrations.quaternion import q_to_euler, q_convert_wxyz  # noqa
from tprmp.envs.gym import Environment # noqa
from tprmp.envs.tasks import PalletizingBoxes # noqa
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 3
dt = 0.01
hidden_dim = 128
num_epoch = 5
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
    if not isinstance(demos, list):
        demos = [demos]
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    # training routine
    print('Start training')
    model.train()  # Set the model to training mode
    for i in tqdm(range(num_epoch)):
        running_loss = deque(maxlen=log_window)
        for demo in demos:
            traj, d_traj, dd_traj = demo['traj'].astype(np.float32), demo['traj_vel'].astype(np.float32), demo['traj_accel'].astype(np.float32)
            for t in range(traj.shape[1]):
                x = np.append(traj[:3, t], q_to_euler(q_convert_wxyz(traj[3:, t])))
                x, dx, ddx = torch.from_numpy(x).unsqueeze(0).to(device), torch.from_numpy(d_traj[:, t]).unsqueeze(0).to(device), torch.from_numpy(dd_traj[:, t]).to(device)
                optimizer.zero_grad()
                M, c, g = model(x, dx)
                rmp = M @ ddx + c + g  # output current RMP
                # geo_term = torch.inverse(M) @ g + torch.norm(dx)
                loss = criterion(rmp, torch.zeros_like(rmp, device=device))  # enforce conservative structure
                running_loss.append(loss.item())
                loss.backward()  # Backprop gradients to all tensors in the network
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_bound)
                optimizer.step()  # Update trainable weights
        scheduler.step()
        if i % log_window == 0:
            print("Epoch {} loss:{}".format(i + 1, np.mean(running_loss)))


def retrieve(model, init_vel, T):
    env = Environment(task=PalletizingBoxes(), disp=True)
    env._ee_vel = init_vel
    for t in np.linspace(0, T, int(T / dt) + 1):
        x = np.append(env.ee_pose[:3], q_to_euler(q_convert_wxyz(env.ee_pose[3:])))
        x, dx = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device), torch.from_numpy(env.ee_vel.astype(np.float32)).unsqueeze(0).to(device)
        ddx = model.rmp(x, dx)
        ddx = ddx.cpu().detach().numpy()
        env.step(ddx)
        time.sleep(dt)


def draw_force_pattern(model):
    x, y, z = np.meshgrid(np.linspace(0, 5, 10), np.linspace(0, 5, 10), np.linspace(0, 5, 10))
    p = np.vstack(map(np.ravel, (x, y, z))).T
    p = np.append(p, np.zeros_like(p), axis=1)
    p = torch.from_numpy(p.astype(np.float32)).to(device)
    v = torch.zeros_like(p)
    M, c, g = model(p, v)
    g = g.cpu().detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x, y, z, g[:, 0], g[:, 1], g[:, 2])
    ax.title('Potential g')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Example run: python train_deeprmp.py test.p')
    parser.add_argument('task', help='The task folder', type=str, default='pick')
    parser.add_argument('data', help='The data file', type=str, default='sample.p')
    args = parser.parse_args()

    TASK_DIR = join(ROOT_DIR, 'data', 'tasks', args.task)
    MODEL_DIR = join(TASK_DIR, 'models')
    DATA_DIR = join(TASK_DIR, 'demos')
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    data_file = join(DATA_DIR, args.data)
    demo = load_demos(data_file)
    # trajs = simple_demo(T, dt)
    # preprocess data
    traj_accel = compute_traj_velocity(demo['traj_vel'], dt)
    traj_accel[:, -2] = traj_accel[:, -3]
    traj_accel[:, -1] = traj_accel[:, -3]
    demo['traj_accel'] = traj_accel
    print('Viewing demos, close to train...')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(demo['traj'][0, :], demo['traj'][1, :], demo['traj'][2, :], marker="o", color="r", markersize=2)
    plt.show()
    model = DeepRMPNetwork(6).to(device)
    train(model, demo)
    torch.save(model, join(MODEL_DIR, str(time.time()) + '.pth'))
    # model = torch.load(join(MODEL_DIR, 'sample.pth')).to(device)
    # draw_force_pattern(model)
    init_vel = demo['traj_vel'][:, 0]
    retrieve(model, init_vel, T)
