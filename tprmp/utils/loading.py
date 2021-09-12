import pickle
import numpy as np
from tprmp.demonstrations.base import Demonstration
from tprmp.demonstrations.quaternion import q_convert_wxyz
from tprmp.demonstrations.manifold import Manifold


def load(demo_file):
    with open(demo_file, 'rb') as f:
        data = pickle.load(f)
    return data


def save_demos(demo_file, trajs, traj_vels, frames, tags, dt=0.01):
    data = {'trajs': trajs, 'traj_vel': traj_vels, 'frames': frames, 'tags': tags, 'dt': dt}
    with open(demo_file, 'wb') as f:
        pickle.dump(data, f)


def load_demos(data_file, smooth=True, tag=None, convert_wxyz=True):
    '''Load data into Demonstration class with format xyzwxyz'''
    data = load(data_file)
    dt = data['dt']
    demos = []
    if convert_wxyz:
        for k, v in data['frames'].items():
            if isinstance(v, list):
                for m in range(len(v)):
                    v[m][3:] = q_convert_wxyz(v[m][3:])  # convert to wxyz
            else:
                v[3:] = q_convert_wxyz(v[3:])
    manifold = Manifold.get_manifold_from_name('R^3 x S^3')
    for m in range(len(data['trajs'])):
        if tag is not None and data['tags'][m] != tag:
            continue
        if convert_wxyz:
            data['trajs'][m][3:] = q_convert_wxyz(data['trajs'][m][3:])
        demo = Demonstration(data['trajs'][m], smooth=smooth, manifold=manifold, dt=dt, tag=data['tags'][m])
        for k, v in data['frames'].items():
            p = v[m] if isinstance(v, list) else v
            demo.add_frame_from_pose(p, k)
        demos.append(demo)
    return demos


def load_demos_2d(data_file, smooth=True, dt=0.01, first=True):
    '''Load 2d demonstrations'''
    data = load(data_file)
    demos = []
    manifold = Manifold.get_euclidean_manifold(2)
    if isinstance(data, list):
        data = np.array(data)
    start_f, end_f = data[0][:, 0], data[0][:, -1]
    for d in data:
        demo = Demonstration(d, manifold=manifold, dt=dt)
        if first:
            demo.add_frame_from_pose(start_f, 'start')
            demo.add_frame_from_pose(end_f, 'end')
        else:
            demo.add_frame_from_pose(d[:, 0], 'start')
            demo.add_frame_from_pose(d[:, -1], 'end')
        demos.append(demo)
    return demos
