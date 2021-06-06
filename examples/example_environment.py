import sys
from os.path import join, dirname, abspath


ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)


from tprmp.envs.tasks import Task, PalletizingBoxes # noqa
from tprmp.envs.gym import Environment  # noqa


task = PalletizingBoxes()
env = Environment(task=task, disp=True)
input()
