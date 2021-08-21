import sys
import argparse
import numpy as np
from os.path import join, dirname, abspath
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load  # noqa
from tprmp.visualization.demonstration import plot_demo  # noqa
from tprmp.models.tp_rmp import TPRMP  # noqa
from tprmp.demonstrations.base import Demonstration  # noqa
from tprmp.demonstrations.manifold import Manifold  # noqa
from tprmp.demonstrations.frame import Frame  # noqa
from tprmp.demonstrations.quaternion import q_convert_wxyz, q_from_euler, q_convert_xyzw  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python test_tprmp.py test.p')
parser.add_argument('task', help='The task folder', type=str, default='pick')
parser.add_argument('data', help='The data file', type=str, default='data.p')
args = parser.parse_args()

DATA_DIR = join(ROOT_DIR, 'data', 'tasks', args.task, 'demos')
data_file = join(DATA_DIR, args.data)
# parameters
T = 300
dt = 0.01
NUM_COMP = 30
alpha, beta = 0., 0.
stiff_scale = 3.
d_min = 0.
d_scale = 3.
energy = 0.
sigma = 1.
var_scale = 1.
r = 0.01
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
# train tprmp
model = TPRMP(num_comp=NUM_COMP, name=args.task, sigma=sigma, stiff_scale=stiff_scale, d_scale=d_scale)
model.train(demos, alpha=alpha, beta=beta, d_min=d_min, energy=energy, var_scale=var_scale)
# model.model.plot_model(demos)
