import sys
import argparse
from os.path import join, dirname, abspath
import matplotlib
import numpy as np
import logging
import time
import pickle
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 16

ROOT_DIR = join(dirname(abspath(__file__)), '..', '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.experiment import Experiment  # noqa

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Example run: python experiment.py 1')
parser.add_argument('--mode', help='Load or not', type=int, default=1)
args = parser.parse_args()

task = 'test'
demo_type = '2D'
demo_names = ['C', 'C1', 'G', 'hat', 'hat1', 'I', 'I1', 'J', 'L', 'L1', 'P', 'S', 'S1', 'S2', 'U']
start_random_radius = 0.01
moving_goal_radius = 0.2
disturb_magnitude = 10.
disturb_period = [50, 150]
disturb = True
v_eps = 5e-2
goal_eps = 0.2
Kp = 1 * np.eye(2)
Kd = 0.1 * np.eye(2)
worker = Experiment(task=task, demo_type=demo_type, demo_names=demo_names, start_random_radius=start_random_radius, moving_goal_radius=moving_goal_radius,
                    disturb_magnitude=disturb_magnitude, disturb_period=disturb_period, v_eps=v_eps, goal_eps=goal_eps)
worker.load_demos()

if args.mode == 1:
    print('Start training models...')
    worker.train()
elif args.mode == 2:
    print('Tracking experiment...')
    error = worker.tracking_experiment()
elif args.mode == 3:
    print('Tracking baseline experiment...')
    error = worker.tracking_baseline_experiment(Kp, Kd, dt=0.005)
elif args.mode == 4:
    print('Adaptation experiment...')
    error = worker.adaptation_experiment(disturb=disturb)
elif args.mode == 5:
    print('Adaptation with increasing disturb force experiment...')
    disturb = True
    test_comps = [9]
    worker.test_comps = test_comps
    disturb_magnitude = 10. * np.arange(1, 10)
    error = {}
    for dm in disturb_magnitude:
        worker.disturb_magnitude = dm
        error[dm] = worker.adaptation_experiment(disturb=disturb)
elif args.mode == 6:
    print('Adaptation with increasing degree of difference experiment...')
    disturb = False
    test_comps = [9]
    worker.test_comps = test_comps
    moving_goal_radius = 0.1 * np.arange(1, 10)
    error = {}
    for r in moving_goal_radius:
        worker.moving_goal_radius = r
        error[r] = worker.adaptation_experiment(disturb=disturb)
elif args.mode == 7:
    print('Adaptation with rmpflow...')
    disturb = False
    num_comp = 9
    worker.max_steps = 2000
    radius = 0.1 * np.arange(1, 10)
    error = {}
    for R in radius:
        error[R] = worker.composable_experiment(9, R, disturb=disturb)
filename = join(worker.experiment_path, 'experiment_' + str(args.mode) + '_' + str(time.time()) + '.p')
with open(filename, 'wb') as f:
    pickle.dump(error, f)
