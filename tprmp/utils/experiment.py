import numpy as np
import os
import logging
from os.path import join, dirname, realpath
from tqdm import tqdm  # Displays a progress bar
import pybullet as p

from tprmp.models.tp_rmp import TPRMP
from tprmp.models.rmp_tree import RMPLeaf, RMPNode, RMPRoot
from tprmp.models.rmp_models import CollisionAvoidance
from tprmp.models.pd import PDController
from tprmp.models.rmp import compute_riemannian_metric
from tprmp.envs.gym import Environment
from tprmp.envs.tasks import PickBox
from tprmp.utils.loading import load_demos, load_demos_2d
from tprmp.demonstrations.base import Demonstration
from tprmp.demonstrations.frame import Frame
from tprmp.demonstrations.manifold import Manifold
from tprmp.demonstrations.quaternion import q_convert_wxyz, q_convert_xyzw

_path_file = dirname(realpath(__file__))
DATA_PATH = join(_path_file, '..', '..', 'data', 'tasks')


class Experiment(object):
    logger = logging.getLogger(__name__)

    def __init__(self, **kwargs):
        self.verbose = kwargs.get('verbose', False)
        self.task = kwargs.get('task', 'test')
        self.demo_names = kwargs.get('demo_names', ['I'])
        self.demo_type = kwargs.get('demo_type', '2D')
        self.tag = kwargs.get('tag', None)
        self.dt = kwargs.get('dt', 0.01)
        self.max_iter = kwargs.get('max_iter', 2000)
        self.max_steps = kwargs.get('max_steps', 1000)
        self.start_random_radius = kwargs.get('start_random_radius', 0.01)
        self.moving_goal_radius = kwargs.get('moving_goal_radius', 0.05)
        self.disturb_magnitude = kwargs.get('disturb_magnitude', 1.)
        self.disturb_period = kwargs.get('disturb_period', [0, 100])
        self.omega = kwargs.get('omega', np.pi)
        self.v_eps = kwargs.get('v_eps', 1e-5)
        self.goal_eps = kwargs.get('goal_eps', 1e-1)
        self.wait = kwargs.get('wait', 10)
        self.goal_frame = kwargs.get('goal_frame', 'end')
        self.experiment_path = kwargs.get('experiment_path', join(DATA_PATH, self.task, 'experiments'))
        self.demo_path = kwargs.get('demo_path', join(DATA_PATH, self.task, 'demos'))
        os.makedirs(self.experiment_path, exist_ok=True)
        # training params
        self.test_comps = kwargs.get('test_comps', [5, 7, 9, 11, 13, 15])
        # self.displace_var = kwargs.get('displace_var', 0.01 * np.arange(10))
        self.stiff_scale = kwargs.get('stiff_scale', 1.)
        self.mass_scale = kwargs.get('mass_scale', 0.5)
        self.var_scale = kwargs.get('var_scale', 3.)
        self.delta = kwargs.get('delta', 2.)
        self.models = []
        self.demos = []
        self.manifold = None

    def load_demos(self):
        for i, n in enumerate(self.demo_names):
            data_file = join(self.demo_path, n + '.p')
            if self.demo_type == '2D':
                self.demos.append(load_demos_2d(data_file, dt=self.dt))
            elif self.demo_type == '6D':
                self.demos.append(load_demos(data_file, tag=self.tag))
            else:
                raise ValueError(f'Demo type {self.demo_type} is unregconized!')
        self.manifold = self.demos[0][0].manifold

    def train(self):
        for i, n in enumerate(tqdm(self.demo_names)):
            for num_comp in self.test_comps:
                model = TPRMP(num_comp=num_comp, name=self.task, stiff_scale=self.stiff_scale, mass_scale=self.mass_scale, var_scale=self.var_scale, delta=self.delta)
                model.train(self.demos[i], max_iter=self.max_iter, verbose=self.verbose)
                model.save(name=n + '_' + str(num_comp) + '.p')

    def init_rmpflow(self, model):
        # environment
        env = Environment(task=PickBox(), disp=True, sampling_hz=int(1 / self.dt))
        env.task.spawn_sphere(env)
        sphere_pose, _ = p.getBasePositionAndOrientation(env.task.sphere_id)
        # build rmp tree
        root = RMPRoot('C_space', manifold=Manifold.get_euclidean_manifold(len(env.joints)))

        def tprmp_psi(q):
            pose = np.array(env.ee_pose)
            pose[3:] = q_convert_wxyz(pose[3:])
            return pose

        def tprmp_J(q):
            if isinstance(q, np.ndarray):
                q = q.tolist()
            zero_v = [0.] * len(q)
            joint_states = (q, zero_v, zero_v)
            J_pos, J_rot = env.compute_ee_jacobian(joint_states=joint_states)
            J = np.append(np.array(J_pos), np.array(J_rot), axis=0)
            return J

        def ws_J(q):
            if isinstance(q, np.ndarray):
                q = q.tolist()
            zero_v = [0.] * len(q)
            joint_states = (q, zero_v, zero_v)
            J_pos, _ = env.compute_ee_jacobian(joint_states=joint_states)
            return np.array(J_pos)

        ws_node = RMPNode('R^3_space', parent=root, manifold=Manifold.get_euclidean_manifold(3), psi=lambda x: env.ee_pose[:3], J=ws_J)
        CollisionAvoidance('CA_space', parent=ws_node, c=np.array(sphere_pose), R=env.task.R)
        RMPLeaf('TPRMP_space', model.rmp, parent=root, manifold=model.model.manifold, psi=tprmp_psi, J=tprmp_J)
        return env, root

    def tracking_experiment(self):
        tracking_error = {}
        for num_comp in tqdm(self.test_comps):
            tracking_error[num_comp] = []
            for i, n in enumerate(self.demo_names):
                name = n + '_' + str(num_comp) + '.p'
                model = TPRMP.load(self.task, model_name=name)
                sample = self.demos[i][0]
                frames = sample.get_task_parameters()
                x0, dx0 = sample.traj[:, 0], sample._d_traj[:, 0]
                traj, success = self.execute(model, frames, x0, dx0)
                tracking_error[num_comp].append(self.mse_criteria(traj, sample.traj))
        return tracking_error

    def tracking_baseline_experiment(self, Kp, Kd, dt=0.01):
        tracking_baseline_error = []
        model = PDController(self.manifold, Kp=Kp, Kd=Kd)
        ratio = int(self.dt / dt)
        for i, n in enumerate(tqdm(self.demo_names)):
            sample = self.demos[i][0]
            x, dx = sample.traj[:, 0], sample._d_traj[:, 0]
            traj = [x]
            model.update_targets(sample.traj[:, 1], np.zeros_like(dx))
            count = 0
            t = 0
            while True:
                t += 1
                if t % ratio == 0:
                    i = int(t / ratio)
                    if i < sample.traj.shape[1] - 1:
                        model.update_targets(sample.traj[:, i + 1], np.zeros_like(dx))
                ddx = model.retrieve(x, dx)
                dx = ddx * self.dt + dx
                x = self.manifold.exp_map(dx * self.dt, base=x)
                traj.append(x)
                if np.linalg.norm(dx) < self.v_eps:
                    count += 1
                    if count >= self.wait:
                        break
                else:
                    count = 0
            traj = np.array(traj).T
            tracking_baseline_error.append(self.mse_criteria(traj, sample.traj))
        return tracking_baseline_error

    def adaptation_experiment(self, disturb=False):
        goal_errors = {}
        for num_comp in tqdm(self.test_comps):
            goal_errors[num_comp] = []
            for i, n in enumerate(self.demo_names):
                name = n + '_' + str(num_comp) + '.p'
                model = TPRMP.load(self.task, model_name=name)
                sample = self.demos[i][0]
                start_pose = sample.traj[:, 0]
                if self.demo_type == '2D':  # only 2D demos has randomized start pose
                    start_pose += np.random.uniform(low=-self.start_random_radius, high=self.start_random_radius) * np.ones_like(start_pose)
                    origin = sample.traj[:, -1]
                elif self.demo_type == '6D':
                    frames = sample.get_task_parameters()
                    origin = frames['obj_frame'].transform(self.manifold.get_origin())
                else:
                    raise ValueError(f'Demo type {self.demo_type} is unregconized!')
                A, b = Demonstration.construct_linear_map(self.manifold, start_pose)
                start_frame = Frame(A, b, manifold=self.manifold)
                x, dx = start_pose, np.zeros_like(sample._d_traj[:, 0])
                t = 0
                moving = True
                count = 0
                while t < self.max_steps:
                    if moving:
                        if self.demo_type == '2D':
                            end_pose = origin + self.moving_goal_radius * np.array([np.cos(self.omega * t * self.dt), np.sin(self.omega * t * self.dt)])
                            A, b = Demonstration.construct_linear_map(self.manifold, end_pose)
                            end_frame = Frame(A, b, manifold=self.manifold)
                            frames = {'start': start_frame, 'end': end_frame}
                        elif self.demo_type == '6D':
                            end_pose = np.array(origin)
                            end_pose[:3] += self.moving_goal_radius * np.array([np.cos(self.omega * t * self.dt), np.sin(self.omega * t * self.dt), 0.])
                            A, b = Demonstration.construct_linear_map(self.manifold, end_pose)
                            end_frame = Frame(A, b, manifold=self.manifold)
                            frames = {'ee_frame': start_frame, 'obj_frame': end_frame}
                        else:
                            raise ValueError(f'Demo type {self.demo_type} is unregconized!')
                    ddx = model.retrieve(x, dx, frames=frames)
                    if disturb and (t >= self.disturb_period[0] and t <= self.disturb_period[1]):
                        M = compute_riemannian_metric(x, model._global_mvns, mass_scale=model._mass_scale)
                        v = dx / np.linalg.norm(dx)
                        if self.demo_type == '2D':
                            df = self.disturb_magnitude * np.array([v[1], v[0]])
                        elif self.demo_type == '6D':
                            df = np.zeros_like(ddx)
                            df[:3] = self.disturb_magnitude * Experiment.perpendicular_vector(v[:3])
                        else:
                            raise ValueError(f'Demo type {self.demo_type} is unregconized!')
                        ddx += np.linalg.inv(M) @ df
                    dx = ddx * self.dt + dx
                    x = self.manifold.exp_map(dx * self.dt, base=x)
                    goal = model._global_mvns[-1].mean
                    d = np.linalg.norm(self.manifold.log_map(x, base=goal))
                    if d < self.goal_eps:
                        moving = False
                    if np.linalg.norm(dx) < self.v_eps:
                        count += 1
                        if count >= self.wait:
                            break
                    else:
                        count = 0
                    t += 1
                goal_errors[num_comp].append(np.linalg.norm(self.manifold.log_map(x, base=goal)))
        return goal_errors

    def composable_experiment(self, num_comp, disturb=False):
        '''Only works with demo_type 6D'''
        goal_errors = []
        for i, n in enumerate(tqdm(self.demo_names)):
            name = n + '_' + str(num_comp) + '.p'
            model = TPRMP.load(self.task, model_name=name)
            manifold = model.model.manifold
            env, root = self.init_rmpflow(model)
            sample = self.demos[i][0]
            start_pose = sample.traj[:, 0]
            A, b = Demonstration.construct_linear_map(manifold, start_pose)
            start_frame = Frame(A, b, manifold=manifold)
            frames = sample.get_task_parameters()
            origin = frames['obj_frame'].transform(self.manifold.get_origin())
            box_id = env.task.box_id
            # init
            curr = np.array(start_pose)
            curr[3:] = q_convert_xyzw(curr[3:])
            env.setp(curr)
            env._ee_pose = curr
            env._config, _, _ = env.get_joint_states(np_array=True)
            t = 0
            moving = True
            count = 0
            while t < self.max_steps:
                if moving:
                    end_pose = np.array(origin)
                    end_pose[:3] += self.moving_goal_radius * np.array([np.cos(self.omega * t * self.dt), np.sin(self.omega * t * self.dt), 0.])
                    A, b = Demonstration.construct_linear_map(self.manifold, end_pose)
                    end_frame = Frame(A, b, manifold=self.manifold)
                    frames = {'ee_frame': start_frame, 'obj_frame': end_frame}
                p.resetBasePositionAndOrientation(box_id, end_pose[:3], q_convert_xyzw(end_pose[3:]))
                model.generate_global_gmm(frames)
                ddq = root.solve(env.config, env.config_vel)
                env.step(ddq, return_data=False, config_space=True)
                goal = model._global_mvns[-1].mean
                d = np.linalg.norm(self.manifold.log_map(env.ee_pose, base=goal))
                if d < self.goal_eps:
                    moving = False
                if np.linalg.norm(env.ee_vel) < self.v_eps:
                    count += 1
                    if count >= self.wait:
                        break
                else:
                    count = 0
                t += 1
            goal_errors.append(np.linalg.norm(self.manifold.log_map(env.ee_pose, base=goal)))
        return goal_errors

    def execute(self, model, frames, x0, dx0):
        x, dx = x0, dx0
        traj = [x]
        model.generate_global_gmm(frames)
        count = 0
        while True:
            ddx = model.retrieve(x, dx)
            dx = ddx * self.dt + dx
            x = self.manifold.exp_map(dx * self.dt, base=x)
            traj.append(x)
            if np.linalg.norm(dx) < self.v_eps:
                count += 1
                if count >= self.wait:
                    break
            else:
                count = 0
        goal = frames[self.goal_frame].transform(self.manifold.get_origin())
        success = np.linalg.norm(self.manifold.log_map(x, base=goal)) < self.goal_eps
        return np.array(traj).T, success

    def mse_criteria(self, traj, demo):
        mse = 0.
        for t in range(traj.shape[1]):
            mse += min([np.linalg.norm(self.manifold.log_map(demo[:, j], base=traj[:, t])) for j in range(demo.shape[1])])
        last_demo_idx = np.argmin([np.linalg.norm(self.manifold.log_map(demo[:, j], base=traj[:, -1])) for j in range(demo.shape[1])])
        for i in range(last_demo_idx, demo.shape[1]):
            mse += np.linalg.norm(self.manifold.log_map(demo[:, i], base=traj[:, -1]))
        return mse / (traj.shape[1] + demo.shape[1] - last_demo_idx)

    @staticmethod
    def perpendicular_vector(v):
        if v[1] == 0 and v[2] == 0:
            if v[0] == 0:
                raise ValueError('zero vector')
            else:
                return np.cross(v, [0, 1, 0])
        return np.cross(v, [1, 0, 0])
