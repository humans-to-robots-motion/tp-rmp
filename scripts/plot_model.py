import sys
import argparse
from os.path import join, dirname, abspath
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load_demos  # noqa
from tprmp.models.tp_hsmm import TPHSMM  # noqa
from tprmp.visualization.demonstration import plot_demo  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python plot_model.py sample.p')
parser.add_argument('task', help='The task folder', type=str, default='test')
parser.add_argument('data', help='The model file', type=str, default='sample.p')
args = parser.parse_args()

DATA_DIR = join(ROOT_DIR, 'data', 'tasks', args.task, 'demos')
data_file = join(DATA_DIR, args.data)
demos = load_demos(data_file)
model = TPHSMM.load(args.task, args.data)
model.plot_model(demos, tagging=False)
