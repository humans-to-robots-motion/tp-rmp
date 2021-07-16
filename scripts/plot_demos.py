import sys
import argparse
from os.path import join, dirname, abspath

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load_demos  # noqa
from tprmp.visualization.demonstration import plot_demo  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python load_demos.py test.p')
parser.add_argument('task', help='The task folder', type=str, default='test')
parser.add_argument('data', help='The data file', type=str, default='data.p')
args = parser.parse_args()

DATA_DIR = join(ROOT_DIR, 'data', 'tasks', args.task, 'demos')
data_file = join(DATA_DIR, args.data)
demos = load_demos(data_file)
plot_demo(demos, only_global=False, plot_quat=False, new_fig=True, new_ax=True, show=True)
