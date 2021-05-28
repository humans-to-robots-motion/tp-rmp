import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.load_demos import load_demos  # noqa
from tprmp.demonstrations.trajectory import compute_traj_derivatives  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python load_demos.py test.p')
parser.add_argument('task', help='The task folder', type=str, default='test')
parser.add_argument('data', help='The data file', type=str, default='data.p')
args = parser.parse_args()

DATA_DIR = join(ROOT_DIR, 'data', 'tasks', args.task, 'demos')
data_file = join(DATA_DIR, args.data)
trajs = load_demos(data_file)
dt = 0.1

for traj in trajs:
    X = np.array(list(zip(traj[0], traj[1]))).T
    X, dX, ddX = compute_traj_derivatives(X, dt)
    plt.quiver(traj[0], traj[1], dX[0], dX[1], color="b", width=0.003, headwidth=2, headlength=3)
    plt.quiver(traj[0], traj[1], ddX[0], ddX[1], color="g", width=0.003, headwidth=2, headlength=3)
for traj in trajs:
    plt.plot(traj[0], traj[1], linestyle="none", marker="o", color="r", markersize=2)
plt.title('Demos')
plt.show()
