import pickle


def load(demo_file):
    with open(demo_file, 'rb') as f:
        data = pickle.load(f)
    return data


def save_demos(demo_file, trajs, traj_vels, frames, tags):
    data = {'trajs': trajs, 'traj_vel': traj_vels, 'frames': frames, 'tags': tags}
    with open(demo_file, 'wb') as f:
        pickle.dump(data, f)
