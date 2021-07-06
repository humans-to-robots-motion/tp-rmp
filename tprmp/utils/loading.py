import pickle
from tprmp.demonstrations.base import Demonstration
from tprmp.demonstrations.quaternion import q_convert_wxyz


def load(demo_file):
    with open(demo_file, 'rb') as f:
        data = pickle.load(f)
    return data


def save_demos(demo_file, trajs, traj_vels, frames, tags, dt=0.01):
    data = {'trajs': trajs, 'traj_vel': traj_vels, 'frames': frames, 'tags': tags, 'dt': dt}
    with open(demo_file, 'wb') as f:
        pickle.dump(data, f)


def load_demos(data_file):
    '''Load data into Demonstration class with format xyzwxyz'''
    data = load(data_file)
    dt = data['dt']
    demos = []
    for k, v in data['frames'].items():
        if isinstance(v, list):
            for m in range(len(v)):
                v[m][3:] = q_convert_wxyz(v[m][3:])
        else:
            v[3:] = q_convert_wxyz(v[3:])
    # convert to wxyz
    for m in range(len(data['trajs'])):
        data['trajs'][m][3:] = q_convert_wxyz(data['trajs'][m][3:])
        demo = Demonstration(data['trajs'][m], dt=dt, tag=data['tags'][m])
        for k, v in data['frames'].items():
            if isinstance(v, list):
                demo.add_frame_from_pose(v[m], k)
            else:
                demo.add_frame_from_pose(v, k)
        demos.append(demo)
    return demos
