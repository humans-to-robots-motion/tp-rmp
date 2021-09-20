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
from tprmp.models.tp_rmp import TPRMP  # noqa
from tprmp.models.rmp_tree import RMPLeaf, RMPNode, RMPRoot  # noqa
from tprmp.models.rmp_models import CollisionAvoidance  # noqa
from tprmp.demonstrations.manifold import Manifold  # noqa
from tprmp.demonstrations.base import Demonstration  # noqa
from tprmp.demonstrations.frame import Frame  # noqa
from tprmp.demonstrations.quaternion import q_convert_wxyz, q_from_euler, q_convert_xyzw  # noqa
from tprmp.envs.gym import Environment # noqa
from tprmp.envs.tasks import PickBox # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python test_tprmp.py')
parser.add_argument('--task', help='The task folder', type=str, default='pick')
parser.add_argument('--data', help='The data file', type=str, default='huber.p')
parser.add_argument('--mode', help='The data file', type=int, default=0)
args = parser.parse_args()

# parameters
N = 2
T = 5
R = 0.06
r = 0.1
obj_v = 0.05 * np.array([1., 1., 0.])
omega = 0.3 * np.pi
cv = 0.3 * np.pi
a = 0.2
moving_goal_radius = 0.15
goal_eps = 0.2
model = TPRMP.load(args.task, model_name=args.data)
manifold = model.model.manifold
dt = model.model.dt
max_steps = 2000
# environment
sphere_origin = np.array([0.5, -0.08, 0.2, 0., 0., 0., 1.])
env = Environment(task=PickBox(R=R), disp=True, sampling_hz=int(1/dt))
env.task.spawn_sphere(env)
c, _ = p.getBasePositionAndOrientation(env.task.sphere_id)
# build rmp tree
root = RMPRoot('C_space', manifold=Manifold.get_euclidean_manifold(len(env.joints)))


def tprmp_psi(q):
    pose = np.array(env.ee_pose)
    pose[3:] = q_convert_wxyz(pose[3:])
    return pose


def tprmp_J(q):
    if isinstance(q, np.ndarray):
        q = q.tolist()
    zero_v = [0.] * len(q)
    joint_states = (q, zero_v, zero_v)
    J_pos, J_rot = env.compute_ee_jacobian(joint_states=joint_states)
    J = np.append(np.array(J_pos), np.array(J_rot), axis=0)
    return J


def ws_J(q):
    if isinstance(q, np.ndarray):
        q = q.tolist()
    zero_v = [0.] * len(q)
    joint_states = (q, zero_v, zero_v)
    J_pos, _ = env.compute_ee_jacobian(joint_states=joint_states)
    return np.array(J_pos)


ws_node = RMPNode('R^3_space', parent=root, manifold=Manifold.get_euclidean_manifold(3), psi=lambda x: env.ee_pose[:3], J=ws_J)
ca_node = CollisionAvoidance('CA_space', parent=ws_node, c=np.array(c), R=R)
tprmp_node = RMPLeaf('TPRMP_space', model.rmp, parent=root, manifold=manifold, psi=tprmp_psi, J=tprmp_J)
# init start & end pose
start_pose = np.array([4.35803125e-1, 1.09041607e-1, 2.90120033e-1, 9.93708392e-1, -1.76660117e-4, 1.11998216e-1, 9.23757958e-6])
A, b = Demonstration.construct_linear_map(manifold, start_pose)
ee_frame = Frame(A, b, manifold=manifold)
box_id = env.task.box_id
obj_rotation = q_convert_xyzw(q_from_euler(np.array([np.pi/2, 0., 0.])))

for _ in range(N):
    env.reset()
    env.task.spawn_sphere(env, sphere_origin)
    moving = True
    origin = np.array([0.56179028, -0.18820972, env.task.box_size[1] / 2]) + np.random.uniform(low=-r, high=r) * np.array([1, 1, 0])
    curr = np.array(start_pose)
    curr[3:] = q_convert_xyzw(curr[3:])
    env.setp(curr)
    env._ee_pose = curr
    env._ee_vel = np.zeros(6)
    env._config, _, _ = env.get_joint_states(np_array=True)
    env._config_vel = np.zeros(6)
    # execution
    for t in np.linspace(0, T, T * env.sampling_hz + 1):
        # update collision object
        sphere_pose = sphere_origin + np.array([0., a * np.cos(cv * t), 0., 0., 0., 0., 1.])
        ca_node.c = sphere_pose[:3]
        p.resetBasePositionAndOrientation(env.task.sphere_id, sphere_pose[:3], sphere_pose[3:])
        if moving:
            if args.mode == 1:
                obj_position = origin + moving_goal_radius * np.array([np.cos(omega * t), np.sin(omega * t), 0.])
            else:
                origin += obj_v * dt
                obj_position = origin
            p.resetBasePositionAndOrientation(box_id, obj_position, obj_rotation)
            obj_pose = np.append(obj_position, q_convert_wxyz(obj_rotation))
            A, b = Demonstration.construct_linear_map(manifold, obj_pose)
            obj_frame = Frame(A, b, manifold=manifold)
            frames = {'ee_frame': ee_frame, 'obj_frame': obj_frame}
            model.generate_global_gmm(frames)
        ddq = root.solve(env.config, env.config_vel)
        env.step(ddq, return_data=False, config_space=True)
        x = env.ee_pose
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
