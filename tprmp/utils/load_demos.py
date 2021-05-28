import pickle


def load_demos(demo_file):
    with open(demo_file, 'rb') as f:
        data = pickle.load(f)
    return data
