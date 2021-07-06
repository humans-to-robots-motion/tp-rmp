import sys
import argparse
from os.path import join, dirname, abspath
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load  # noqa
from tprmp.demonstrations.base import Demonstration  # noqa
from tprmp.demonstrations.quaternion import q_convert_wxyz  # noqa
from tprmp.models.tp_hsmm import TPHSMM  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python load_demos.py test.p')
parser.add_argument('task', help='The task folder', type=str, default='test')
parser.add_argument('data', help='The data file', type=str, default='data.p')
args = parser.parse_args()

DATA_DIR = join(ROOT_DIR, 'data', 'tasks', args.task, 'demos')
data_file = join(DATA_DIR, args.data)
data = load(data_file)
dt = 0.01
# convert to wxyz
traj = data['traj']
traj[3:] = q_convert_wxyz(traj[3:])

demo = Demonstration(data['traj'], dt=dt, tag='pick-top')
for k, v in data['frames'].items():
    v[3:] = q_convert_wxyz(v[3:])
    demo.add_frame_from_pose(v, k)

model = TPHSMM(num_comp=5, name=args.task)
gamma = model.train(demo, plot=True)
model.plot_model(demo)
model.save(name='sample')
