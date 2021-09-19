import sys
import argparse
import numpy as np
from os.path import join, dirname, abspath
import matplotlib.pyplot as plt
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load_demos_2d  # noqa
from tprmp.visualization.demonstration import plot_frame_2d  # noqa
from tprmp.visualization.dynamics import plot_dissipation_field, plot_potential_field, visualize_rmp # noqa
from tprmp.models.tp_rmp import TPRMP  # noqa
from tprmp.demonstrations.base import Demonstration  # noqa
from tprmp.demonstrations.frame import Frame  # noqa
from tprmp.models.rmp import compute_riemannian_metric  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python test_tprmp.py test.p')
parser.add_argument('--task', help='The task folder', type=str, default='test')
parser.add_argument('--mode', help='Background', type=int, default=1)
parser.add_argument('--demo', help='The data file', type=str, default='hat.p')
parser.add_argument('--data', help='The data file', type=str, default='hat_13.p')
args = parser.parse_args()

DATA_DIR = join(ROOT_DIR, 'data', 'tasks', args.task, 'demos')
data_file = join(DATA_DIR, args.demo)
# parameters
limits = [0., 4.5]
res = 0.2
colormap = 'RdBu' if args.mode == 1 else 'YlOrBr'
dt = 0.01
start_random_radius = 0.01
moving_goal_radius = 0.2
omega = np.pi
disturb = False
disturb_period = [50, 150]
disturb_magnitude = 10.
goal_eps = 0.2
v_eps = 5e-2
wait = 10
# load data
demos = load_demos_2d(data_file, dt=dt)
# train tprmp
sample = demos[0]
manifold = sample.manifold
frames = sample.get_task_parameters()
model = TPRMP.load(args.task, model_name=args.data)
# execution
start_pose = sample.traj[:, 0]
start_pose += np.random.uniform(low=-start_random_radius, high=start_random_radius) * np.ones_like(start_pose)
A, b = Demonstration.construct_linear_map(manifold, start_pose)
start_frame = Frame(A, b, manifold=manifold)
origin = sample.traj[:, -1]
x, dx = start_pose, np.zeros_like(sample._d_traj[:, 0])
t = 0
moving = True
count = 0
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_xlim([limits[0], limits[1]])
ax.set_ylim([limits[0], limits[1]])
line, = ax.plot(x[0], x[1], marker="o", color="b", markersize=1, linestyle='None', alpha=1.)
X, Y = np.meshgrid(np.arange(limits[0], limits[1], res), np.arange(limits[0], limits[1], res))
Z = np.zeros_like(X)
vmax = (model.phi0[0] + 20) if args.mode == 1 else model.d0[-1]
mesh = ax.pcolormesh(X, Y, Z, cmap=colormap, shading='auto', vmin=0., vmax=vmax, alpha=0.5)
fig.colorbar(mesh, ax=ax)
frame_line = {'start': [None, None], 'end': [None, None]}
for f in frame_line:
    frame_line[f][0], = ax.plot([], [], color="r", alpha=1.)
    frame_line[f][1], = ax.plot([], [], color="g", alpha=1.)
traj = [x]
while True:
    if moving:
        end_pose = origin + moving_goal_radius * np.array([np.cos(omega * t * dt), np.sin(omega * t * dt)])
        A, b = Demonstration.construct_linear_map(manifold, end_pose)
        end_frame = Frame(A, b, manifold=manifold)
    frames = {'start': start_frame, 'end': end_frame}
    ddx = model.retrieve(x, dx, frames=frames)
    if disturb and (t >= disturb_period[0] and t <= disturb_period[1]):
        M = compute_riemannian_metric(x, model._global_mvns, mass_scale=model._mass_scale)
        v = dx / np.linalg.norm(dx)
        df = disturb_magnitude * np.array([v[1], v[0]])
        ddx += np.linalg.inv(M) @ df
    dx = ddx * dt + dx
    x = dx * dt + x
    goal = model._global_mvns[-1].mean
    d = np.linalg.norm(manifold.log_map(x, base=goal))
    if d < goal_eps:
        moving = False
    if np.linalg.norm(dx) < v_eps:
        count += 1
        if count >= wait:
            break
    else:
        count = 0
    t += 1
    # plotting
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if args.mode == 1:
                Z[i, j] = model.compute_potential_field(np.array([X[i, j], Y[i, j]]))
            else:
                Z[i, j] = model.compute_dissipation_field(np.array([X[i, j], Y[i, j]]))
    mesh.update({'array': Z.ravel()})
    fx_x, fx_y = [], []
    fy_x, fy_y = [], []
    for f, v in frames.items():
        A = v.A
        b = v.b
        x_goal = b[:2] + A[:2, 0] * 0.05
        y_goal = b[:2] + A[:2, 1] * 0.05
        frame_line[f][0].set_xdata([b[0], x_goal[0]])
        frame_line[f][0].set_ydata([b[1], x_goal[1]])
        frame_line[f][1].set_xdata([b[0], y_goal[0]])
        frame_line[f][1].set_ydata([b[1], y_goal[1]])
    traj.append(x)
    line.set_xdata(np.array(traj)[:, 0])
    line.set_ydata(np.array(traj)[:, 1])
    fig.canvas.draw()
    fig.canvas.flush_events()
