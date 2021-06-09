import sys
import numpy as np
import pybullet as p
import time
import os
from os.path import join, dirname, abspath


ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)

from tprmp.utils.loading import save_demos # noqa
from tprmp.envs.tasks import PalletizingBoxes # noqa
from tprmp.envs.gym import Environment  # noqa

env = Environment(task=PalletizingBoxes(), disp=True, real_time_step=True)
margin = 0.001
speed = 0.01
timeout = 20
direct = False

save_path = join(ROOT_DIR, 'data', 'tasks', 'pick', 'demos')
os.makedirs(save_path, exist_ok=True)
save_name = join(save_path, str(time.time()) + '.p')
# grasp box
box_id = env.task.goals[0][0][0]
target = p.getBasePositionAndOrientation(box_id)
pos = np.array(target[0])
pos[2] += env.task.box_size[2] / 2 + margin
pose = np.append(pos, target[1])
res = env.record_trajectory()
env.movep(pose, speed=speed, timeout=timeout, direct=direct)
traj, traj_vel = res.result()
env.movep(env.home_pose, speed=speed, timeout=timeout, direct=direct)
for t in range(traj.shape[1]):
    env.movep(traj[:, t], speed=speed, timeout=timeout, direct=True)
    time.sleep(1 / env.sampling_hz)
frames = {'obj_frame': target, 'ee_frame': env.home_pose}
save_demos(save_name, traj, traj_vel, frames)
