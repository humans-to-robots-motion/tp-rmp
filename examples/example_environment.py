import sys
import numpy as np
from os.path import join, dirname, abspath


ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)


from tprmp.envs.tasks import Task, PalletizingBoxes # noqa
from tprmp.envs.gym import Environment  # noqa


task = PalletizingBoxes()
env = Environment(task=task, disp=True)
# test compute jacobian
J_pos, J_rot = env.compute_ee_jacobian()
J_pos, J_rot = np.array(J_pos), np.array(J_rot)
print(J_pos)
print(J_rot)
input()
