import sys
import argparse
import numpy as np
from os.path import join, dirname, abspath
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
from tprmp.visualization.demonstration import plot_demo, _equalize_axes  # noqa
from tprmp.visualization.dynamics import plot_potential_force_components, plot_weight_map # noqa
from tprmp.visualization.models import _plot_gmm_global # noqa
from tprmp.models.tp_rmp import TPRMP  # noqa
from tprmp.demonstrations.base import Demonstration  # noqa
from tprmp.demonstrations.manifold import Manifold  # noqa

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
var_scale = 2.
res = 0.01
max_z = 1000
margin = 0.5
verbose = False
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
sample = demos[0]
frames = sample.get_task_parameters()
model = TPRMP.load(args.task, model_name=args.data)
x = np.array([2., 2.5])
plot_weight_map(model.model, frames, margin=margin, res=res, new_fig=True, show=False)
plot_potential_force_components(model, frames, x, show=True)
