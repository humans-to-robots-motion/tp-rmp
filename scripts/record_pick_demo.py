import sys
import numpy as np
import pybullet as p
import time
import argparse
import os
from os.path import join, dirname, abspath


ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)

from tprmp.utils.loading import save_demos # noqa
from tprmp.envs.tasks import PalletizingBoxes # noqa
from tprmp.envs.gym import Environment  # noqa
from tprmp.demonstrations.base import Demonstration  # noqa
from tprmp.demonstrations.frame import Frame  # noqa
from tprmp.demonstrations.manifold import Manifold  # noqa
from tprmp.demonstrations.quaternion import q_convert_wxyz, q_convert_xyzw, q_from_euler  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python record_pick_demo.py test.p')
parser.add_argument('-n', help='Name of record', type=str, default=str(time.time()))
parser.add_argument('--num', help='Num of demos for each skill', type=bool, default=3)
args = parser.parse_args()

env = Environment(task=PalletizingBoxes(), disp=True, real_time_step=True)
margin = 0.0009
speed = 0.01
timeout = 10
direct = False

save_path = join(ROOT_DIR, 'data', 'tasks', 'pick', 'demos')
os.makedirs(save_path, exist_ok=True)
save_name = join(save_path, args.n + '.p')
manifold = Manifold.get_manifold_from_name('R^3 x S^3')
box_id = env.task.goals[0][0][0]
r = 0.1
trajs = []
traj_vels = []
tags = []
obj_frames = []
# grasp top
pose_mean = np.array([0.5, -0.25, env.task.box_size[2] / 2])
for _ in range(args.num):
    theta = np.random.random() * 2 * np.pi
    rotation = q_convert_xyzw(q_from_euler(np.array([0., 0., theta])))
    position = pose_mean + np.random.uniform(low=-r, high=r) * np.array([1, 1, 0])
    p.resetBasePositionAndOrientation(box_id, position, rotation)
    target = p.getBasePositionAndOrientation(box_id)
    pose = np.append(target[0], target[1])
    pose[2] += env.task.box_size[2] / 2 + margin
    res = env.record_trajectory()
    env.movep(pose, speed=speed, timeout=timeout, direct=direct)
    traj, traj_vel = res.result()
    env.movep(env.home_pose, speed=speed, timeout=timeout, direct=direct)
    trajs.append(traj)
    traj_vels.append(traj_vel)
    tags.append('pick_top')
    obj_frames.append(np.append(target[0], target[1]))
# grasp side
pose_mean = np.array([0.5, -0.25, env.task.box_size[0] / 2])
for _ in range(args.num):
    theta = np.random.random() * 2 * np.pi
    rotation = q_convert_xyzw(q_from_euler(np.array([np.pi/2, 0., 0.])))
    position = pose_mean + np.random.uniform(low=-r, high=r) * np.array([1, 1, 0])
    p.resetBasePositionAndOrientation(box_id, position, rotation)
    target = p.getBasePositionAndOrientation(box_id)
    obj_pose = np.append(target[0], target[1])
    obj_pose[3:] = q_convert_wxyz(obj_pose[3:])
    A, b = Demonstration.construct_linear_map(manifold, obj_pose)
    obj_frame = Frame(A, b, manifold)
    pose = obj_frame.transform(np.append([0., 0., 0.], q_from_euler(np.array([-np.pi/2, 0., 0.]))))
    pose[3:] = q_convert_xyzw(pose[3:])
    pose[2] += env.task.box_size[0] / 2 + margin
    res = env.record_trajectory()
    env.movep(pose, speed=speed, timeout=timeout, direct=direct)
    traj, traj_vel = res.result()
    env.movep(env.home_pose, speed=speed, timeout=timeout, direct=direct)
    trajs.append(traj)
    traj_vels.append(traj_vel)
    tags.append('pick_side')
    obj_frames.append(np.append(target[0], target[1]))
frames = {'obj_frame': obj_frames, 'ee_frame': env.home_pose}
save_demos(save_name, trajs, traj_vels, frames, tags)
