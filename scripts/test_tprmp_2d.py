import sys
import argparse
import numpy as np
from os.path import join, dirname, abspath, expanduser
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load_demos_2d  # noqa
from tprmp.models.tp_rmp import TPRMP  # noqa
from tprmp.visualization.demonstration import plot_frame_2d  # noqa
from tprmp.models.rmp import compute_riemannian_metric  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python test_tprmp.py test.p')
parser.add_argument('--task', help='The task folder', type=str, default='test')
parser.add_argument('--mode', help='Background', type=int, default=1)
parser.add_argument('--demo', help='The data file', type=str, default='U.p')
parser.add_argument('--data', help='The data file', type=str, default='U_9.p')
args = parser.parse_args()

DATA_DIR = join(ROOT_DIR, 'data', 'tasks', args.task, 'demos')
data_file = join(DATA_DIR, args.demo)
fps = 30
video_file = join(expanduser("~"), 'U_tracking.mp4')
metadata = dict(artist='Matplotlib')
writer = FFMpegWriter(fps=fps, metadata=metadata)
# parameters
limits = [0., 4.5]
res = 0.05
colormap = 'RdBu' if args.mode == 1 else 'YlOrBr'
dt = 0.01
disturb = False
disturb_period = [50, 150]
disturb_magnitude = 10.
v_eps = 1e-2
max_steps = 2000
wait = 10
# load data
demos = load_demos_2d(data_file, dt=dt)
# train tprmp
sample = demos[0]
manifold = sample.manifold
frames = sample.get_task_parameters()
model = TPRMP.load(args.task, model_name=args.data)
model.generate_global_gmm(frames)
x, dx = sample.traj[:, 0], sample._d_traj[:, 0]
traj = [x]
t = 0
plt.ion()
fig = plt.figure()
writer.setup(fig, outfile=video_file, dpi=None)
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_xlim([limits[0], limits[1]])
ax.set_ylim([limits[0], limits[1]])
line, = ax.plot(x[0], x[1], marker="o", color="b", markersize=1, linestyle='None', alpha=1.)
ax.plot(sample.traj[0], sample.traj[1], color="b", linestyle='--', alpha=0.6)
plot_frame_2d(frames.values())
X, Y = np.meshgrid(np.arange(limits[0], limits[1], res), np.arange(limits[0], limits[1], res))
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if args.mode == 1:
            Z[i, j] = model.compute_potential_field(np.array([X[i, j], Y[i, j]]))
        else:
            Z[i, j] = model.compute_dissipation_field(np.array([X[i, j], Y[i, j]]))
mesh = ax.pcolormesh(X, Y, Z, cmap=colormap, shading='auto', vmin=0., vmax=Z.max(), alpha=0.5)
fig.colorbar(mesh, ax=ax)
while t < max_steps:
    ddx = model.retrieve(x, dx)
    if disturb and (t >= disturb_period[0] and t <= disturb_period[1]):
        M = compute_riemannian_metric(x, model._global_mvns, mass_scale=model._mass_scale)
        v = dx / np.linalg.norm(dx)
        df = disturb_magnitude * np.array([v[1], v[0]])
        ddx += np.linalg.inv(M) @ df
    dx = ddx * dt + dx
    x = dx * dt + x
    t += 1
    if np.linalg.norm(dx) < v_eps:
        break
    # plotting
    traj.append(x)
    line.set_xdata(np.array(traj)[:, 0])
    line.set_ydata(np.array(traj)[:, 1])
    fig.canvas.draw()
    fig.canvas.flush_events()
    writer.grab_frame()
writer.finish()
