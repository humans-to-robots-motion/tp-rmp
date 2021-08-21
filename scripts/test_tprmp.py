import sys
import argparse
import numpy as np
import pybullet as p
from os.path import join, dirname, abspath
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load_demos  # noqa
from tprmp.visualization.demonstration import plot_demo  # noqa
from tprmp.models.tp_rmp import TPRMP  # noqa
from tprmp.demonstrations.base import Demonstration  # noqa
from tprmp.demonstrations.frame import Frame  # noqa
from tprmp.demonstrations.quaternion import q_convert_wxyz, q_from_euler, q_convert_xyzw  # noqa
from tprmp.envs.gym import Environment # noqa
from tprmp.envs.tasks import PalletizingBoxes # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python test_tprmp.py test.p')
parser.add_argument('task', help='The task folder', type=str, default='pick')
parser.add_argument('data', help='The data file', type=str, default='data.p')
args = parser.parse_args()

DATA_DIR = join(ROOT_DIR, 'data', 'tasks', args.task, 'demos')
data_file = join(DATA_DIR, args.data)
# parameters
T = 300
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
demos = load_demos(data_file, tag='pick_side')
demos.pop(0)
manifold = demos[0].manifold
# plot_demo(demos, only_global=False, plot_quat=False, new_fig=True, new_ax=True, show=True)
# train tprmp
model = TPRMP(num_comp=NUM_COMP, name=args.task, sigma=sigma, stiff_scale=stiff_scale, d_scale=d_scale)
model.train(demos, alpha=alpha, beta=beta, d_min=d_min, energy=energy, var_scale=var_scale)
model.model.plot_model(demos)
# test tprmp
env = Environment(task=PalletizingBoxes(), disp=True)
ee_pose = np.array(env.home_pose)
ee_pose[3:] = q_convert_wxyz(ee_pose[3:])
# env._ee_pose[1] -= 0.05
# env.setp(env.ee_pose)
A, b = Demonstration.construct_linear_map(manifold, ee_pose)
ee_frame = Frame(A, b, manifold=manifold)
box_id = env.task.goals[0][0][0]
position = np.array([0.5, -0.25, env.task.box_size[0] / 2]) + np.random.uniform(low=-r, high=r) * np.array([1, 1, 0])
rotation = q_convert_xyzw(q_from_euler(np.array([np.pi/2, 0., 0.])))
p.resetBasePositionAndOrientation(box_id, position, rotation)
target = p.getBasePositionAndOrientation(box_id)
obj_pose = np.append(position, q_convert_wxyz(rotation))
A, b = Demonstration.construct_linear_map(manifold, obj_pose)
obj_frame = Frame(A, b, manifold=manifold)
frames = {'ee_frame': ee_frame, 'obj_frame': obj_frame}
model.generate_global_gmm(frames)
for t in np.linspace(0, T, T * env.sampling_hz + 1):
    x = np.append(env.ee_pose[:3], q_convert_wxyz(env.ee_pose[3:]))
    dx = env.ee_vel
    ddx = model.retrieve(x, dx, frames)
    env.step(ddx)
    # input()
