import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load_demos_2d  # noqa
from tprmp.visualization.demonstration import plot_demo  # noqa
from tprmp.visualization.dynamics import plot_dissipation_field, plot_potential_field, visualize_rmp # noqa
from tprmp.models.tp_rmp import TPRMP  # noqa
from tprmp.demonstrations.base import Demonstration  # noqa
from tprmp.demonstrations.frame import Frame  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python test_tprmp.py test.p')
parser.add_argument('--task', help='The task folder', type=str, default='test')
parser.add_argument('--data', help='The data file', type=str, default='test3.p')
parser.add_argument('--plot', help='The data file', type=str, default='dissipative')
args = parser.parse_args()
# parameters
T = 200
dt = 0.01
res = 0.05
max_z = 1000
margin = 0.5
start_v = 0.
end_v = 0.6
limits = [0., 4.]
model = TPRMP.load(args.task, model_name=args.data)
manifold = model.model.manifold
start_pose = np.array([1.4, 1.1])
end_pose = np.array([3.1, 2.6])
plt.figure()
for t in range(T):
    start_pose += start_v * dt
    end_pose += end_v * dt
    A, b = Demonstration.construct_linear_map(manifold, start_pose)
    start_frame = Frame(A, b, manifold=manifold)
    A, b = Demonstration.construct_linear_map(manifold, end_pose)
    end_frame = Frame(A, b, manifold=manifold)
    frames = {'start': start_frame, 'end': end_frame}
    if args.plot == 'potential':
        plot_potential_field(model, frames, margin=margin, max_z=max_z, three_d=True, res=res, limits=limits, new_fig=False, show=False)
    elif args.plot == 'dissipative':
        plot_dissipation_field(model, frames, margin=margin, res=res, limits=limits, new_fig=False, show=False)
    plt.draw()
    plt.pause(0.00001)
    plt.cla()
