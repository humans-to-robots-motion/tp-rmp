import os
import pickle
import time

_path_file = os.path.dirname(os.path.realpath(__file__))


class Recorder2D:
    def __init__(self, plot, task_name='traj'):
        self.save_path = os.path.join(_path_file, '..', '..', 'data', 'demonstrations')
        os.makedirs(self.save_path, exist_ok=True)
        self.save_name = os.path.join(self.save_path, task_name + str(time.time()) + '.p')
        self.plot = plot
        self.trajs = []
        self.curr_traj_x = []
        self.curr_traj_y = []
        # for drawing purpose
        self.xs = []
        self.ys = []
        self.cid = plot.figure.canvas.mpl_connect('button_press_event', self)
        self.cid = plot.figure.canvas.mpl_connect('key_press_event', self.save)

    def save(self, event):
        if event.key == 'e':
            with open(self.save_name, 'wb') as f:
                pickle.dump(self.trajs, f)

    def __call__(self, event):
        if event.inaxes != self.plot.axes:
            return
        if event.button == 1:
            self.curr_traj_x.append(event.xdata)
            self.curr_traj_y.append(event.ydata)
        elif event.button == 3:
            self.curr_traj_x.append(event.xdata)
            self.curr_traj_y.append(event.ydata)
            self.trajs.append((self.curr_traj_x, self.curr_traj_y))
            self.curr_traj_x = []
            self.curr_traj_y = []
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.plot.set_data(self.xs, self.ys)
        self.plot.figure.canvas.draw()
