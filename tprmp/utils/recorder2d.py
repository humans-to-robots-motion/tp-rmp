import os
import numpy as np
from os.path import join, dirname, realpath
from tprmp.demonstrations.trajectory import compute_traj_velocity, smooth_traj
import pickle
import time

_path_file = dirname(realpath(__file__))


class Recorder2D:
    def __init__(self, plot, task_name='test', N=3, var=0.1, smooth=True, window=5):
        # N must be odd
        assert N % 2 == 1
        assert window % 2 == 1
        self.save_path = join(_path_file, '..', '..', 'data', 'tasks', task_name, 'demos')
        os.makedirs(self.save_path, exist_ok=True)
        self.save_name = join(self.save_path, task_name + str(time.time()) + '.p')
        self.plot = plot
        self.trajs = []
        self.curr_traj = []
        self.drawing = False
        self.N = N
        self.var = var
        self.smooth = smooth
        self.halfw = int((window - 1) / 2)
        # for drawing purpose
        self.xs = []
        self.ys = []
        plot.figure.canvas.mpl_connect('button_press_event', self.on_press)
        plot.figure.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        plot.figure.canvas.mpl_connect('key_press_event', self.save)

    def save(self, event):
        if event.key == 'e':
            if len(self.trajs) == 1:
                if self.smooth:
                    self.trajs[0] = smooth_traj(self.trajs[0])
                traj = self.trajs[0]
                add_trajs = [np.zeros_like(traj) for _ in range(self.N - 1)]
                r = np.linspace(-self.var, self.var, self.N).tolist()
                r.pop(int((self.N - 1) / 2))
                r = np.array(r)
                tau = int(round(traj.shape[1] / 4))
                scheduler = np.linspace(1., 0., tau)
                dtraj = compute_traj_velocity(traj, 1.)
                vs = np.zeros_like(dtraj)
                for t in range(traj.shape[1] - 1):
                    vs[:, t] = np.sum(dtraj[:, max(0, t - self.halfw):(t + self.halfw + 1)], axis=1) / (self.halfw * 2 + 1)
                    vs[:, t] /= np.linalg.norm(vs[:, t])
                for t in range(traj.shape[1] - 1):
                    ov = np.array([-vs[1, t], vs[0, t]])
                    if t < traj.shape[1] - tau:
                        for i, n in enumerate(r):
                            add_trajs[i][:, t] = traj[:, t] + ov * n
                    else:
                        j = t - traj.shape[1] + tau
                        for i, n in enumerate(scheduler[j] * r):
                            add_trajs[i][:, t] = traj[:, t] + ov * n
                for i in range(self.N - 1):
                    add_trajs[i][:, -1] = traj[:, -1]
                self.trajs.extend(add_trajs)
            with open(self.save_name, 'wb') as f:
                pickle.dump(self.trajs, f)

    def mouse_move(self, event):
        if event.inaxes != self.plot.axes:
            return
        if self.drawing:
            self.curr_traj.append([event.xdata, event.ydata])
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)

    def on_press(self, event):
        if self.drawing:
            self.drawing = False
            self.trajs.append(np.array(self.curr_traj).T)
            self.curr_traj = []
            self.plot.set_data(self.xs, self.ys)
            self.plot.figure.canvas.draw()
        else:
            self.drawing = True
