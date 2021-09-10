import sys
import argparse
import numpy as np
from os.path import join, dirname, abspath
import matplotlib.pyplot as plt
import matplotlib
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 16

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load  # noqa
from tprmp.visualization.demonstration import plot_demo, _equalize_axes, _plot_traj_frames  # noqa
from tprmp.visualization.dynamics import _plot_dissipation_field_global, _plot_potential_field_global  # noqa
from tprmp.visualization.models import _plot_gmm_global, _plot_gmm_frames # noqa
from tprmp.models.tp_rmp import TPRMP  # noqa
from tprmp.demonstrations.base import Demonstration  # noqa
from tprmp.demonstrations.manifold import Manifold  # noqa
from tprmp.demonstrations.frame import Frame  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python test_tprmp.py test.p')
parser.add_argument('--loading', help='Load or not', type=bool, default=True)
parser.add_argument('--task', help='The task folder', type=str, default='test')
parser.add_argument('--demo', help='The data file', type=str, default='test3.p')
parser.add_argument('--data', help='The data file', type=str, default='test3.p')
args = parser.parse_args()

DATA_DIR = join(ROOT_DIR, 'data', 'tasks', args.task, 'demos')
data_file = join(DATA_DIR, args.demo)
# parameters
dt = 0.01
NUM_COMP = 5
alpha, beta = 0., 0.
stiff_scale = 1.
mass_scale = 1.
tau = 0.5
delta = 2.
potential_method = 'huber'
train_method = 'match_energy'
d_min = 0.
d_scale = 1.
energy = 0.
var_scale = 1.
res = 0.05
max_z = 1000
margin = 0.5
verbose = False
limits = [0., 4.]
# load data
data = load(data_file)
demos = []
manifold = Manifold.get_euclidean_manifold(2)
for d in data:
    traj = np.stack(d)
    demo = Demonstration(traj, manifold=manifold, dt=dt)
    demo.add_frame_from_pose(traj[:, 0], 'start')
    demo.add_frame_from_pose(traj[:, -1], 'end')
    demos.append(demo)
plot_demo(demos, only_global=True, plot_quat=False, new_fig=True, new_ax=True, three_d=False, margin=margin, limits=limits, legend=False, show=False)
model = TPRMP.load(args.task, model_name=args.data)
# model = TPRMP(num_comp=NUM_COMP, name=args.task, stiff_scale=stiff_scale, mass_scale=mass_scale, var_scale=var_scale, tau=tau, delta=delta, potential_method=potential_method, d_scale=d_scale)
# model.train(demos, alpha=alpha, beta=beta, d_min=d_min, train_method=train_method, energy=energy, verbose=verbose)
# plt.figure()
# axs = _plot_traj_frames(demos, legend=False, plot_quat=False, three_d=False, plot_frames=True, margin=margin)
# _plot_gmm_frames(model.model, model.model.frame_names, axs=axs, three_d=False)
mid = [sum(limits) / 2., sum(limits) / 2.]
ranges = (limits[1] - limits[0]) / 2.
# start_pose = np.array([0.5, 0.5])
# end_pose = np.array([3., 2.])
# A, b = Demonstration.construct_linear_map(manifold, start_pose)
# start_frame = Frame(A, b, manifold=manifold)
# A, b = Demonstration.construct_linear_map(manifold, end_pose)
# end_frame = Frame(A, b, manifold=manifold)
# frames = {'start': start_frame, 'end': end_frame}
frames = demos[0].get_task_parameters()
frames['start']._b += np.array([-0.5, 0.5])
frames['end']._b += np.array([-0.5, 1.])
# plt.figure()
# ax = plt.subplot(111)
# ax.set_aspect('equal')
# ax.set_xlim([limits[0], limits[1]])
# ax.set_ylim([limits[0], limits[1]])
# _plot_gmm_global(model.model, frames, plot_frames=True, three_d=False, legend=False, new_ax=False)
plt.figure()
_plot_potential_field_global(model, frames, mid, ranges, plot_gaussian=True, max_z=max_z, three_d=True, res=res)
plt.figure()
_plot_dissipation_field_global(model, frames, mid, ranges, plot_gaussian=True, res=res)
plt.show()
