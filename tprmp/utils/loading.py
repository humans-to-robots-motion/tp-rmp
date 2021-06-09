import pickle


def load_demos(demo_file):
    with open(demo_file, 'rb') as f:
        data = pickle.load(f)
    return data


def save_demos(demo_file, traj, traj_vel, frames):
    data = {'traj': traj, 'traj_vel': traj_vel, 'frames': frames}
    with open(demo_file, 'wb') as f:
        pickle.dump(data, f)
