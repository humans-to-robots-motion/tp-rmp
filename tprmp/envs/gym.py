from os.path import join, dirname, realpath
import tempfile
import logging
import gym
import numpy as np
import pybullet as p
import time

from tprmp.demonstrations.manifold import Manifold
from tprmp.demonstrations.quaternion import q_convert_wxyz, q_convert_xyzw
from tprmp.utils.threading import threaded

_path_file = dirname(realpath(__file__))
ASSETS_PATH = join(_path_file, '..', '..', 'data', 'assets')
UR5_URDF_PATH = join(ASSETS_PATH, 'ur5', 'ur5.urdf')
UR5_WORKSPACE_URDF_PATH = join(ASSETS_PATH, 'ur5', 'workspace.urdf')
PLANE_URDF_PATH = join(ASSETS_PATH, 'plane', 'plane.urdf')

NUM_JOINTS = 6
NUM_LINKS = 10
VEL_LIMIT = 10
ACCEL_LIMIT = 5


class Environment(gym.Env):
    """OpenAI Gym-style environment class."""
    logger = logging.getLogger(__name__)

    def __init__(self, **kwargs):
        """Creates OpenAI Gym-style environment with PyBullet.
            Parameters
            ----------
            :param task: the task to use. If None, the user must call set_task for the
             environment to work properly.
            :param disp: show environment with PyBullet's built-in display viewer.
            :param shared_memory: run with shared memory.
            :param manifold: manifold of task space
            :param sampling_hz: Sampling freq.
            Raises:
            -------
            RuntimeError: if pybullet cannot load fileIOPlugin.
        """
        task = kwargs.get('task', None)
        disp = kwargs.get('disp', False)
        shared_memory = kwargs.get('shared_memory', False)
        self.sampling_hz = kwargs.get('sampling_hz', 100)
        self.real_time_step = kwargs.get('real_time_step', False)
        self.manifold = kwargs.get('manifold', None)
        if self.manifold is None:
            self.manifold = Manifold.get_manifold_from_name('R^3 x S^3')  # 6-DoFs
        self.home_joint = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0], dtype=np.float32) * np.pi
        self.moving = False  # this flag for recording trajectory
        self.observation_space = gym.spaces.Dict({
            'ee_pose':  # position and normalized quaternion
            gym.spaces.Box(low=np.array([0., 0., 0., -1., -1., -1., -1.], dtype=np.float32),
                           high=np.array([5., 5., 5., 1., 1., 1., 1.], dtype=np.float32),
                           shape=(7,),
                           dtype=np.float32),
            'ee_vel':
            gym.spaces.Box(low=-VEL_LIMIT * np.ones(6, dtype=np.float32),
                           high=VEL_LIMIT * np.ones(6, dtype=np.float32),
                           shape=(6,),
                           dtype=np.float32),
            'config':  # working with UR5
            gym.spaces.Box(low=-np.ones(6, dtype=np.float32) * np.pi,
                           high=np.ones(6, dtype=np.float32) * np.pi,
                           shape=(6,),
                           dtype=np.float32),
            'config_vel':
            gym.spaces.Box(low=-VEL_LIMIT * np.ones(6, dtype=np.float32),
                           high=VEL_LIMIT * np.ones(6, dtype=np.float32),
                           shape=(6,),
                           dtype=np.float32),
            'ft':
            gym.spaces.Discrete(6)  # TODO: implement FT sensor later
        })
        self.position_bounds = gym.spaces.Box(low=np.array([0., 0., 0.], dtype=np.float32),
                                              high=np.array([5., 5., 5.], dtype=np.float32),
                                              shape=(3,),
                                              dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-ACCEL_LIMIT * np.ones(6, dtype=np.float32),
                                           high=ACCEL_LIMIT * np.ones(6, dtype=np.float32),
                                           shape=(6,),
                                           dtype=np.float32)
        # Start PyBullet.
        disp_option = p.DIRECT
        if disp:
            disp_option = p.GUI
            if shared_memory:
                disp_option = p.SHARED_MEMORY
        client = p.connect(disp_option)
        file_io = p.loadPlugin('fileIOPlugin', physicsClientId=client)
        if file_io < 0:
            raise RuntimeError('pybullet: cannot load FileIO!')
        if file_io >= 0:
            p.executePluginCommand(file_io,
                                   textArgument=ASSETS_PATH,
                                   intArgs=[p.AddFileIOAction],
                                   physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setAdditionalSearchPath(ASSETS_PATH)
        p.setAdditionalSearchPath(tempfile.gettempdir())
        p.setTimeStep(1. / self.sampling_hz)
        # If using --disp, move default camera closer to the scene.
        if disp:
            if self.real_time_step:
                p.setRealTimeSimulation(1)
            target = p.getDebugVisualizerCamera()[11]
            p.resetDebugVisualizerCamera(cameraDistance=1.1,
                                         cameraYaw=90,
                                         cameraPitch=-25,
                                         cameraTargetPosition=target)
        if task:
            self.set_task(task)  # TODO: implement set_task function
            self.reset()

    def get_joint_states(self):
        joint_states = p.getJointStates(self.ur5, self.joints)
        j = [state[0] for state in joint_states]
        j_vel = [state[1] for state in joint_states]
        j_torque = [state[3] for state in joint_states]
        return j, j_vel, j_torque

    def compute_ee_jacobian(self):
        j, j_vel, j_torque = self.get_joint_states()
        return p.calculateJacobian(self.ur5, self.ee_tip, np.zeros(3), j, j_vel, j_torque)

    def set_task(self, task):
        self.task = task

    def add_object(self, urdf, pose, category='rigid'):
        """List of (fixed, rigid, or deformable) objects in env. Assuming abs urdf path"""
        fixed_base = 1 if category == 'fixed' else 0
        obj_id = p.loadURDF(urdf, pose[:3], pose[3:], useFixedBase=fixed_base)
        self.obj_ids[category].append(obj_id)
        return obj_id

    def movej(self, targj, speed=0.01, timeout=5, direct=False):
        """Move UR5 to target joint configuration."""
        self.moving = True
        if direct:  # should work with RealTimeSimulation
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(bodyIndex=self.ur5,
                                        jointIndices=self.joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=targj,
                                        positionGains=gains)
            self.moving = False
            return True
        else:
            t0 = time.time()
            while (time.time() - t0) < timeout:
                currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
                currj = np.array(currj)
                diffj = targj - currj
                if all(np.abs(diffj) < 1e-2):
                    self.moving = False
                    return True
                # Move with constant velocity  #TODO:  implement variated velocity if needed
                norm = np.linalg.norm(diffj)
                v = diffj / norm if norm > 0 else 0
                stepj = currj + v * speed
                gains = np.ones(len(self.joints))
                p.setJointMotorControlArray(bodyIndex=self.ur5,
                                            jointIndices=self.joints,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPositions=stepj,
                                            positionGains=gains)
                if not self.real_time_step:
                    p.stepSimulation()
            Environment.logger.warn(f'movej exceeded {timeout} second timeout. Skipping.')
            self.moving = False
            return False

    def movep(self, pose, speed=0.01, timeout=5, direct=False):
        """Move UR5 to target end effector pose."""
        targj = self.solve_ik(pose)
        return self.movej(targj, speed, timeout, direct)

    def setp(self, pose):
        """This should work with p.stepSimulation()"""
        targj = self.solve_ik(pose)
        for i in range(len(self.joints)):
            p.resetJointState(self.ur5, self.joints[i], targj[i])
        if not self.real_time_step:
            p.stepSimulation()

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5,
            endEffectorLinkIndex=self.ee_tip,
            targetPosition=pose[:3],
            targetOrientation=pose[3:],
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.home_joint).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def reset(self):
        """Performs common reset functionality for all supported tasks."""
        if not self.task:
            raise ValueError('environment task must be set. Call set_task or pass '
                             'the task arg in the environment constructor.')
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setGravity(0, 0, -9.8)
        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.loadURDF(PLANE_URDF_PATH, [0, 0, -0.001])
        p.loadURDF(UR5_WORKSPACE_URDF_PATH, [0.5, 0, 0])
        # Load UR5 robot arm equipped with suction end effector.
        self.ur5 = p.loadURDF(UR5_URDF_PATH)
        self.ee = self.task.ee(self.ur5, NUM_LINKS - 1, self.obj_ids)
        self.ee_tip = NUM_LINKS  # Link ID of suction cup.
        # Get revolute joint indices of robot (skip fixed joints).
        n_joints = p.getNumJoints(self.ur5)
        joints = [p.getJointInfo(self.ur5, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]
        # Move robot to home joint configuration.
        for i in range(len(self.joints)):
            p.resetJointState(self.ur5, self.joints[i], self.home_joint[i])
        # Reset end effector.
        self.ee.release()
        # Reset task.
        self.task.reset(self)
        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        obs, _, _, _ = self.step()
        self.home_pose = obs['ee_pose']
        # a hack to bypass computing Jacobian at step(), these variables are perfect state tracking
        self._ee_pose = obs['ee_pose']
        self._ee_vel = np.zeros(6, dtype=np.float32)
        return obs

    def step(self, action=None):
        """Execute action with acceleration.
            Parameters
            ----------
            :param action: action to execute.
            Returns:
            --------
            (obs, reward, done, info) tuple containing MDP step data.
        """
        if action is None:
            return self.robot_state, 0, False, {}
        ee, ee_vel = self.ee_pose, self.ee_vel  # NOTE: self.ee_vel is a hack to bypass computing Jacobian
        ee_vel += action / self.sampling_hz
        ee_wxyz = np.append(ee[:3], q_convert_wxyz(ee[3:]))
        ee_wxyz = self.manifold.exp_map(ee_vel, base=ee_wxyz)
        ee = np.append(ee_wxyz[:3], q_convert_xyzw(ee_wxyz[3:]))
        self.setp(ee)
        self._ee_pose = ee
        self._ee_vel = ee_vel
        # Get task rewards.
        reward, info = self.task.reward() if action is not None else (0, {})
        done = self.task.done()
        # Add ground truth robot state into info.
        info.update(self.info)
        obs = self.robot_state
        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        # Render only the color image from the first camera.
        # Only support rgb_array for now.
        if mode != 'rgb_array':
            raise NotImplementedError('Only rgb_array implemented')
        color, _, _ = self.render_camera(self.agent_cams[0])
        return color

    def render_camera(self, config):
        """Render RGB-D image with specified camera configuration."""
        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config['position'] + lookdir
        focal_len = config['intrinsics'][0]
        znear, zfar = config['zrange']
        viewm = p.computeViewMatrix(config['position'], lookat, updir)
        fovh = (config['image_size'][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi
        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config['image_size'][1] / config['image_size'][0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)
        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=config['image_size'][1],
            height=config['image_size'][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # Get color image.
        color_image_size = (config['image_size'][0], config['image_size'][1],
                            4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config['noise']:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, config['image_size']))
            color = np.uint8(np.clip(color, 0, 255))
        # Get depth image.
        depth_image_size = (config['image_size'][0], config['image_size'][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
        depth = (2. * znear * zfar) / depth
        if config['noise']:
            depth += self._random.normal(0, 0.003, depth_image_size)
        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)
        return color, depth, segm

    @threaded
    def record_trajectory(self):
        '''This function works with self.movep()'''
        traj = []
        traj_vel = []
        # wait for robot to move
        while not self.moving:
            pass
        while self.moving:
            state = self.robot_state
            traj.append(state['ee_pose'])
            traj_vel.append(state['ee_vel'])
            # time.sleep(1 / self.sampling_hz)
        traj = np.vstack(traj).T
        traj_vel = np.vstack(traj_vel).T
        # record traj at self.sampling_hz
        traj = traj[:, ::self.sampling_hz]
        traj_vel = traj_vel[:, ::self.sampling_hz]
        return traj, traj_vel

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [np.linalg.norm(p.getBaseVelocity(i)[0]) for i in self.obj_ids['rigid']]
        return all(np.array(v) < 5e-3)

    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""
        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = p.getBasePositionAndOrientation(obj_id)
                dim = p.getVisualShapeData(obj_id)[0][3]
                info[obj_id] = (pos, rot, dim)
        return info

    @property
    def robot_state(self):
        ee_state = p.getLinkState(self.ur5, self.ee_tip, computeLinkVelocity=True, computeForwardKinematics=True)  # index of gripper is NUM_LINKS
        ee = np.array(ee_state[0] + ee_state[1])  # [x, y, z, x, y, z, w]
        ee_vel = np.array(ee_state[6] + ee_state[7])  # ee velocity
        j, j_vel, _ = self.get_joint_states()
        state = {'ee_pose': ee, 'ee_vel': ee_vel, 'config': j, 'config_vel': j_vel}
        return state

    @property
    def ee_pose(self):
        return self._ee_pose

    @property
    def ee_vel(self):
        return self._ee_vel
