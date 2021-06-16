import numpy as np
import logging
from scipy.linalg import block_diag

from tprmp.demonstrations.quaternion import q_to_rotation_matrix
from tprmp.demonstrations.frame import Frame
from tprmp.demonstrations.manifold import Manifold
from tprmp.demonstrations.trajectory import compute_traj_derivatives


class Demonstrations(object):
    logger = logging.getLogger(__name__)

    def __init__(self, traj, manifold=None, dt=0.1, smooth=False):
        """
        Parameters
        ----------
        :param traj: np.array of shape (dim, length) or list of the points (np.array of shape (dim,)), contains the
                     points of the trajectory in the global reference frame.

        Optional parameters
        -------------------
        :param dt: float, the sampling time with which the demonstration was recorded
        """
        self._dim_M, self._length = traj.shape
        if manifold is None:
            manifold = Manifold.get_euclidean_manifold(self.dim)
        self._manifold = manifold
        if self._dim_M != manifold.dim_M:
            raise RuntimeError('[Demonstrations] Trajectory dim_M %s and the manifold %s does not match.' % (self._dim_M, manifold.dim_M))
        self._smooth = smooth
        self._task_parameters = dict()
        self._dt = dt
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

    def add_frame_from_linear_map(self, A, b, name, constant=False):
        """
        This method adds a new task parameter to the demonstration for a given linear map.

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
        :param constant: boolean flag that specifies if the task parameter is constant (True) or if it is changing over time (False).
        """
        frame = []
        if constant:
            frame.append(Frame(A, b, self.manifold))
        else:
            if len(A) != len(b) or len(A) != self.length:
                raise RuntimeError("A and b must be lists of length %s if constant is False." % self.length)
            for t in range(len(A)):
                frame.append(Frame(A[t], b[t], self.manifold))
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
            Demonstrations.logger.warn('This frame name already exists. The frame has been overwritten.')
        self._task_parameters[name] = frame
        if self._traj_in_frames is None:
            self.traj_in_frames  # just to invoke compute traj in frames
        else:
            traj, d_traj, dd_traj = self._pullback_traj(name)
            self._traj_in_frames[name] = {'traj': traj, 'd_traj': d_traj, 'dd_traj': dd_traj}

    def create_frame_from_obj_pose(self, pose):
        """
        Create single frame from the corresponding object pose.

        Parameters
        ----------
        :param pose: np.array([x,y,z, w,x,y,z]).

        Returns
        ----------
        :return frame (Frame object): with the correponded manifold of this demonstration.
        """
        A, b = self.construct_linear_map(pose)
        frame = Frame(A, b, manifold=self._manifold)
        return frame

    def construct_linear_map(self, pose):
        """
        Construct orientation matrix A and translation vector b from pose.

        Parameters
        ----------
        :param pose: np.array([x,y,z, w,x,y,z]).

        Returns
        ----------
        :return A (np.array): rotation in tangent space
        :return b (np.array): translation in manifold space
        """
        if self._manifold.name == 'R^3 x S^3':
            q_rot_mat = q_to_rotation_matrix(pose[3:])
            A = block_diag(q_rot_mat, np.eye(3))
            b = pose
        elif 'S^3' not in self._manifold.name and 'R^' in self._manifold.name:  # pure Euclidean
            A = np.eye(self._manifold.dim_M)
            b = pose
        else:
            Demonstrations.logger.warn(f'Manifold name {self._manifold.name} is not recognized! Return no linear map.')
            return
        return A, b

    def _pullback_traj(self, f_name):
        if f_name not in self._task_parameters:
            raise RuntimeError("[Demonstrations] Frame %s not in task parameters!" % f_name)
        if len(self._task_parameters[f_name]) == 1:  # constant frame
            transformed_traj = self._task_parameters[f_name][0].pullback(self.traj)
            transformed_d_traj = self._task_parameters[f_name][0].pullback_tangent(self._d_traj)
            transformed_dd_traj = self._task_parameters[f_name][0].pullback_tangent(self._dd_traj)
        else:  # time-varying frame
            transformed_traj = np.array(self.dim, self.length)
            for t in range(self._length):
                current_frame = self.get_task_parameters(t, f_name)
                transformed_traj[:, t] = current_frame.pullback(self.traj[:, t])
                transformed_d_traj = current_frame.pullback_tangent(self._d_traj[:, t])
                transformed_dd_traj = current_frame.pullback_tangent(self._dd_traj[:, t])
        return transformed_traj, transformed_d_traj, transformed_dd_traj

    @property
    def traj(self):
        return self._traj

    @traj.setter
    def traj(self, value):
        if isinstance(value, list):
            value = np.array(value).T
        traj, d_traj, dd_traj = compute_traj_derivatives(value, dt=self.dt, manifold=self.manifold, smooth=self.smooth)
        self._traj = traj
        self._d_traj = d_traj
        self._dd_traj = dd_traj
        self._dim_M, self._length = self._traj.shape
        if self._dim_M != self._manifold.dim_M:
            raise RuntimeError('[Demonstrations] Trajectory dim_M %s and the manifold %s does not match.' % (self._dim_M, self._manifold.dim_M))
        self._traj_in_frames = None

    @property
    def dt(self):
        return self._dt

    @property
    def dim_M(self):
        return self._dim_M

    @property
    def length(self):
        return self._length

    @property
    def manifold(self):
        return self._manifold

    @property
    def smooth(self):
        return self._smooth

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
                traj, d_traj, dd_traj = self._pullback_traj(f_name)
                self._traj_in_frames[f_name] = {'traj': traj, 'd_traj': d_traj, 'dd_traj': dd_traj}
        return self._traj_in_frames


if __name__ == '__main__':
    from tprmp.demonstrations.quaternion import q_from_euler
    a = 1
    omega = np.pi / 4
    dt = 0.01
    manifold = Manifold.get_manifold_from_name('R^3 x S^3')
    t = np.array(range(0, 80, 1))
    traj_accel = -a * omega**2 * np.outer(np.ones(6), np.cos(omega * t))
    traj_vel = np.cumsum(traj_accel, axis=1) * dt
    traj = [np.array([0, 0, 0, 1, 0, 0, 0])]
    for i in t:
        traj.append(manifold.exp_map(traj_vel[:, i], base=traj[-1]))
    traj = np.vstack(traj).T
    obj_pose = np.append(np.array([1, 1, 1]), q_from_euler(np.array([0, 0, np.pi/2])))
    demo = Demonstrations(traj, manifold=manifold, dt=dt)
    obj_frame = demo.create_frame_from_obj_pose(obj_pose)
    demo.add_frame(obj_frame, 'obj')
    obj_traj = obj_frame.pullback(traj)
    obj_traj, obj_traj_vel, obj_traj_accel = compute_traj_derivatives(obj_traj, dt, manifold)
    print(obj_traj_vel[:, 0])
    print(demo.traj_in_frames['obj']['d_traj'][:, 0])
