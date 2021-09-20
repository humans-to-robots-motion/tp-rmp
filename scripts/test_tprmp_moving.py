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
from tprmp.models.tp_rmp import TPRMP  # noqa
from tprmp.demonstrations.base import Demonstration  # noqa
from tprmp.demonstrations.frame import Frame  # noqa
from tprmp.demonstrations.quaternion import q_convert_wxyz, q_from_euler, q_convert_xyzw  # noqa
from tprmp.envs.gym import Environment # noqa
from tprmp.envs.tasks import PickBox # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python test_tprmp.py')
parser.add_argument('--loading', help='Load or not', type=bool, default=True)
parser.add_argument('--task', help='The task folder', type=str, default='pick')
parser.add_argument('--demo', help='The data file', type=str, default='sample.p')
parser.add_argument('--data', help='The data file', type=str, default='huber.p')
parser.add_argument('--mode', help='The data file', type=int, default=0)
args = parser.parse_args()

DATA_DIR = join(ROOT_DIR, 'data', 'tasks', args.task, 'demos')
data_file = join(DATA_DIR, args.demo)
# parameters
N = 5
T = 5
obj_v = 0.05 * np.array([1., 1., 0.])
omega = 0.3 * np.pi
moving_goal_radius = 0.15
goal_eps = 0.1
NUM_COMP = 10
alpha, beta = 0., 0.
stiff_scale = 1.
mass_scale = 0.2
tau = 0.5
delta = 3.
potential_method = 'huber'
train_method = 'match_energy'
d_min = 0.
d_scale = 1.
energy = 0.
var_scale = 3.
res = 0.01
margin = 0.05
verbose = False
r = 0.1
# load data
demos = load_demos(data_file, tag='pick_side')
dt = demos[0].dt
sampling_hz = int(1. / demos[0].dt)
manifold = demos[0].manifold
# plot_demo(demos, only_global=False, plot_quat=False, new_fig=True, new_ax=True, show=True)
# train tprmp
if args.loading:
    model = TPRMP.load(args.task, model_name=args.data)
else:
    model = TPRMP(num_comp=NUM_COMP, name=args.task, stiff_scale=stiff_scale, mass_scale=mass_scale, var_scale=var_scale, tau=tau, delta=delta, potential_method=potential_method, d_scale=d_scale)
    model.train(demos, alpha=alpha, beta=beta, train_method=train_method, d_min=d_min, energy=energy, verbose=verbose)
    model.save(name=args.data)
# model.model.plot_model(demos, var_scale=1.)
# test tprmp
env = Environment(task=PickBox(), disp=True, sampling_hz=sampling_hz)
ee_pose = np.array(demos[0].traj[:, 0])
A, b = Demonstration.construct_linear_map(manifold, ee_pose)
ee_frame = Frame(A, b, manifold=manifold)
box_id = env.task.box_id
obj_rotation = q_convert_xyzw(q_from_euler(np.array([np.pi/2, 0., 0.])))
start_pose = np.array(demos[0].traj[:, 0])
start_pose[3:] = q_convert_xyzw(start_pose[3:])
for _ in range(N):
    env.reset()
    moving = True
    origin = np.array([0.5, -0.25, env.task.box_size[1] / 2]) + np.random.uniform(low=-r, high=r) * np.array([1, 1, 0])
    env.setp(start_pose)
    env._ee_pose = start_pose
    phi = 0
    for t in np.linspace(0, T, T * env.sampling_hz + 1):
        if moving:
            if args.mode == 1:
                obj_position = origin + moving_goal_radius * np.array([np.cos(omega * t + phi), np.sin(omega * t + phi), 0.])
            else:
                origin += obj_v * dt
                obj_position = origin
            p.resetBasePositionAndOrientation(box_id, obj_position, obj_rotation)
            obj_pose = np.append(obj_position, q_convert_wxyz(obj_rotation))
            A, b = Demonstration.construct_linear_map(manifold, obj_pose)
            obj_frame = Frame(A, b, manifold=manifold)
            frames = {'ee_frame': ee_frame, 'obj_frame': obj_frame}
        x = np.append(env.ee_pose[:3], q_convert_wxyz(env.ee_pose[3:]))
        dx = env.ee_vel
        ddx = model.retrieve(x, dx, frames=frames)
        env.step(ddx)
        if np.linalg.norm(x[:3] - obj_position) < goal_eps:
            moving = False
        if np.linalg.norm(x[:3] - obj_position - np.array([0., 0., env.task.box_size[1] / 2])) < 3e-2:
            grasp_position = x[:3] - np.array([0., 0., env.task.box_size[1] / 2 - 0.0005])
            p.resetBasePositionAndOrientation(box_id, grasp_position, obj_rotation)
            p.stepSimulation()
            env.ee.activate()
            break
    env.movep(env.home_pose, speed=0.0001, timeout=5.)
    env.ee.release()
    input()
