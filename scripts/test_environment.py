import sys
import numpy as np
import pybullet as p
from os.path import join, dirname, abspath


ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)


from tprmp.envs.tasks import PickBox # noqa
from tprmp.envs.gym import Environment  # noqa


max_steps = 1000
env = Environment(task=PickBox(), disp=True)
env.task.spawn_sphere(env)
t = 0
a = 100.
T = 500
timeout = 20
omega = 2 * np.pi / T
margin = 0.0009
speed = 0.00005

# oscillate on y-axis one T
while t < max_steps:
    action = -a * omega**2 * np.array([0., 1., 0., 0., 0., 0.]) * np.cos(omega * t)
    obs, reward, done, info = env.step(action)
    t += 1

# grasp box
box_id = env.task.box_id
target = p.getBasePositionAndOrientation(box_id)
pos = np.array(target[0])
pos[2] += env.task.box_size[2] / 2 + margin
target = np.append(pos, target[1])
env.movep(target, speed=speed, timeout=timeout)
env.ee.activate()
env.movep(env.home_pose, speed=speed, timeout=timeout)
env.ee.release()
for _ in range(100):  # to simulate dropping
    p.stepSimulation()
input()
