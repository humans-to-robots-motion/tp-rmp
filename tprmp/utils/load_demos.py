import pickle

from tprmp.demonstrations.base import Demonstration


def load_demos(demo_file):
    with open(demo_file, 'rb') as f:
        data = pickle.load(f)
