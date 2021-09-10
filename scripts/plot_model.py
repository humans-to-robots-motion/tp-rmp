import sys
import argparse
from os.path import join, dirname, abspath
import logging
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 16

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load_demos  # noqa
from tprmp.models.tp_hsmm import TPHSMM  # noqa
from tprmp.visualization.demonstration import _plot_traj_global, _plot_traj_frames, _equalize_axes  # noqa
from tprmp.visualization.models import _plot_gmm_frames, _plot_gmm_global  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python plot_model.py sample.p')
parser.add_argument('task', help='The task folder', type=str, default='test')
parser.add_argument('data', help='The model file', type=str, default='sample2.p')
parser.add_argument('--tag', help='The model file', type=str, default='pick_side')
args = parser.parse_args()

margin = 0.
DATA_DIR = join(ROOT_DIR, 'data', 'tasks', args.task, 'demos')
data_file = join(DATA_DIR, args.data)
demos = load_demos(data_file, tag=args.tag)
model = TPHSMM(num_comp=6, name=args.task)
model.train(demos)
plt.figure()
_plot_traj_global(demos, legend=False, plot_quat=False, margin=margin)
plt.figure()
axs = _plot_traj_frames(demos, legend=False, plot_quat=False, margin=margin)
_plot_gmm_frames(model, model.frame_names, axs=axs)
plt.figure()
ax = plt.subplot(111, projection="3d")
_equalize_axes(ax, demos[0].traj, margin=margin)
_plot_gmm_global(model, demos[0].get_task_parameters(), plot_frames=True, legend=False, new_ax=False)
plt.show()
