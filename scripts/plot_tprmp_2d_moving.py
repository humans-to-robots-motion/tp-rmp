import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath, expanduser
from matplotlib.animation import FFMpegWriter
import matplotlib
import logging
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 16
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load_demos_2d  # noqa
from tprmp.visualization.dynamics import plot_dissipation_field, plot_potential_field # noqa
from tprmp.models.tp_rmp import TPRMP  # noqa
from tprmp.demonstrations.base import Demonstration  # noqa
from tprmp.demonstrations.frame import Frame  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python test_tprmp.py test.p')
parser.add_argument('--task', help='The task folder', type=str, default='test')
parser.add_argument('--demo', help='The data file', type=str, default='U.p')
parser.add_argument('--data', help='The data file', type=str, default='U_5.p')
parser.add_argument('--plot', help='The data file', type=str, default='dissipative')
args = parser.parse_args()

DATA_DIR = join(ROOT_DIR, 'data', 'tasks', args.task, 'demos')
data_file = join(DATA_DIR, args.demo)
fps = 30
video_file = join(expanduser("~"), 'U.mp4')
metadata = dict(artist='Matplotlib')
writer = FFMpegWriter(fps=fps, metadata=metadata)
# parameters
T = 100
a = 0.5
omega = np.pi
dt = 0.01
res = 0.05
max_z = 1000
margin = 0.5
start_v = 0.
end_v = 0.6
limits = [0., 4.]
model = TPRMP.load(args.task, model_name=args.data)
manifold = model.model.manifold
demos = load_demos_2d(data_file, dt=dt)
sample = demos[0]
start_pose = sample.traj[:, 0]
end_pose = sample.traj[:, -1]
fig = plt.figure()
writer.setup(fig, outfile=video_file, dpi=None)
for t in range(T):
    start_p = start_pose + np.array([a * np.cos(omega * t * dt), 0.])
    end_p = end_pose + np.array([0., a * np.cos(omega * t * dt)])
    A, b = Demonstration.construct_linear_map(manifold, start_p)
    start_frame = Frame(A, b, manifold=manifold)
    A, b = Demonstration.construct_linear_map(manifold, end_p)
    end_frame = Frame(A, b, manifold=manifold)
    frames = {'start': start_frame, 'end': end_frame}
    if args.plot == 'potential':
        plot_potential_field(model, frames, margin=margin, max_z=max_z, three_d=False, res=res, limits=limits, new_fig=False, show=False)
    elif args.plot == 'dissipative':
        plot_dissipation_field(model, frames, margin=margin, res=res, limits=limits, new_fig=False, show=False)
    plt.draw()
    writer.grab_frame()
    plt.pause(0.00001)
    plt.cla()
writer.finish()
