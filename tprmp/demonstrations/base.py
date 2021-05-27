import numpy as np

from tprmp.demonstrations.frame import Frame
from tprmp.demonstrations.trajectory import compute_traj_derivatives


class Demonstration(object):
    def __init__(self, traj, sampling_time=0.1, smooth=False):
        """
        Parameters
        ----------
        :param traj: np.array of shape (dim, length) or list of the points (np.array of shape (dim,)), contains the
                     points of the trajectory in the global reference frame.

        Optional parameters
        -------------------
        :param sampling_time: float, the sampling time with which the demonstration was recorded
        TODO: add manifolds
        """
        self._smooth = smooth
        self._task_parameters = dict()
        self._sampling_time = sampling_time
        self.traj = traj

    def get_task_parameters(self, t=0, frame=None):
        """
        Returns the task parameters at a given time instant t.

        Optional parameters
        -------------------
        :param t: int, the specified time.
        :param frame: str, key of a specific desired task parameter. If None all task_parameters are returned as dict.

        Returns
        -------
        :return frames: Frame or a dict of Frame object.
        """
        if frame is None:
            frames = {}
            for f_key in self._task_parameters:
                frames[f_key] = self._task_parameters[f_key][min(t, len(self._task_parameters[f_key]) - 1)]
        else:
            frames = self._task_parameters[frame][min(t, len(self._task_parameters[frame]) - 1)]
        return frames

    def add_frame_from_A_b(self, A, b, name, constant=False):
        """
        This method adds a new task parameter to the demonstration for a given orientation A and translation b.

        Parameters
        ----------
        :param A: A = [A[0], A[1], ..., A[T]]  list of orientation matrices A[t] of shape (self.dim, self.dim)
                    for each point in time with T=self.length if flag_constant==False. OR:
                  A = orientation matrix of shape (self.dim, self.dim) if flag_constant==True
        :param b: b = [b[0], b[1], ... , b[T]]  list of origin vector b[t] of shape (self.dim,) for each point in
                    time with T=self.length if flag_constant==False. OR:
                  b = origin vector of shape (self.dim,) if flag_constant==True
        :param name: str, the name of the frame

        Optional parameters
        -------------------
        :param constant: boolean flag that specifies if the task parameter is constant (True) or if it is
                    changing over time (False).
        """
        frame = []
        if constant:
            frame.append(Frame(A, b))
        else:
            if len(A) != len(b) or len(A) != self.length:
                raise RuntimeError("A and b must be lists of length %s if constant is False." % self.length)
            for t in range(len(A)):
                frame.append(Frame(A[t], b[t]))
        self.add_frame(frame, name)

    def add_frame(self, frame, name):
        """
        This method adds a new task parameter to the demonstration for a given Frame object or list of Frame objects if
        task parameter is time varying.

        Parameters
        ----------
        :param frame: = [f1, f2, ..., fT] list of Frame objects for each time instant. Or [fc] list of single Frame or
                            fc single Frame if task parameter is constant in time.
        :param name: str, the name of the frame
        """
        if isinstance(frame, Frame):
            frame = [frame]
        if name in self._task_parameters:
            Warning("[Demonstration] This frame name already exists. The frame has been overwritten.")
        self._task_parameters[name] = frame
        if self._traj_in_frames is None:
            self.traj_in_frames  # just to invoke compute traj in frames
        else:
            traj = self._transform_traj(name)
            traj, d_traj, dd_traj = compute_traj_derivatives(traj, dt=self.sampling_time, smooth=self.smooth)
            # NOTE: this is inefficient computation for now. TODO: we should be able to transform velocity and accelerations between frames
            self._traj_in_frames[name] = {'traj': traj, 'd_traj': d_traj, 'dd_traj': dd_traj}

    def _transform_traj(self, f_name):
        if f_name not in self._task_parameters:
            raise RuntimeError("[Demonstration] Frame %s not in task parameters!" % f_name)
        if len(self._task_parameters[f_name]) == 1:  # constant frame
            transformed_traj = self._task_parameters[f_name][0].pullback(self.traj)
        else:  # time-varying frame
            transformed_traj = np.array(self.dim, self.length)
            for t in range(self._length):
                current_frame = self.get_task_parameters(t, f_name)
                transformed_traj[:, t] = current_frame.pullback(self.traj[:, t])
        return transformed_traj

    @property
    def traj(self):
        return self._traj

    @traj.setter
    def traj(self, value):
        if isinstance(value, list):
            value = np.array(value).T
        traj, d_traj, dd_traj = compute_traj_derivatives(value, dt=self.sampling_time, smooth=self.smooth)
        self._traj = traj
        self._d_traj = d_traj
        self._dd_traj = dd_traj
        self._dim_M, self._length = self._traj.shape
        self._traj_in_frames = None

    @property
    def sampling_time(self):
        return self._sampling_time

    @property
    def dim_M(self):
        return self._dim_M

    @property
    def length(self):
        return self._length

    @property
    def nb_frames(self):
        return len(self._task_parameters)

    @property
    def frame_names(self):
        return list(self._task_parameters.keys())

    @property
    def traj_in_frames(self):
        if self._traj_in_frames is None:
            self._traj_in_frames = dict()
            for f_name in self._task_parameters:
                traj = self._transform_traj(f_name)
                traj, d_traj, dd_traj = compute_traj_derivatives(traj, dt=self.sampling_time, smooth=self.smooth)
                # NOTE: this is inefficient computation for now. TODO: we should be able to transform velocity and accelerations between frames
                self._traj_in_frames[f_name] = {'traj': traj, 'd_traj': d_traj, 'dd_traj': dd_traj}
        return self._traj_in_frames
