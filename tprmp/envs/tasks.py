import os
from os.path import join, dirname, realpath, split
import numpy as np
import pybullet as p
import string
import random
import tempfile
import copy
import logging

from tprmp.envs.grippers import Suction
from tprmp.demonstrations.quaternion import q_convert_xyzw, q_convert_wxyz, q_from_euler, q_mul, q_inverse

_path_file = dirname(realpath(__file__))
ASSETS_PATH = join(_path_file, '..', '..', 'data', 'assets')
BOX_URDF = join(ASSETS_PATH, 'box', 'box-template.urdf')
PALLET_URDF = join(ASSETS_PATH, 'pallet', 'pallet.urdf')


class Task():
    """Base Task class."""
    def __init__(self):
        self.ee = Suction
        self.goals = []
        self.progress = 0
        self._rewards = 0

    def reset(self, env):
        self.goals = []
        self.progress = 0  # Task progression metric in range [0, 1].
        self._rewards = 0  # Cumulative returned rewards.

    def reward(self):
        """Get delta rewards for current timestep."""
        # TODO: Implement reward function if needed
        reward, info = 0, {}
        return reward, info

    def done(self):
        # TODO: Implement done function if needed
        return (len(self.goals) == 0) or (self._rewards > 0.99)

    def is_match(self, pose0, pose1, eps=1e-3):  # TODO: test this
        """Check if pose0 and pose1 match within a threshold."""
        # Get translational error.
        diff_pos = pose0[:3].astype(np.float32) - pose1[:3].astype(np.float32)
        dist_pos = np.linalg.norm(diff_pos)
        # Get rotational error
        diff_q = q_mul(q_convert_wxyz(pose1[3:]), q_convert_wxyz(q_inverse(pose0[3:])))
        dist_q = np.linalg.norm(np.abs(diff_q) - np.array([1., 0., 0., 0.]))
        return (dist_pos < eps) and (dist_q < eps)

    def fill_template(self, template, replace):
        """Read a file and replace key strings."""
        with open(template, 'r') as file:
            fdata = file.read()
        for field in replace:
            for i in range(len(replace[field])):
                fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))
        alphabet = string.ascii_lowercase + string.digits
        rname = ''.join(random.choices(alphabet, k=16))
        tmpdir = tempfile.gettempdir()
        template_filename = split(template)[-1]
        fname = join(tmpdir, f'{template_filename}.{rname}')
        with open(fname, 'w') as file:
            file.write(fdata)
        return fname

    def get_random_size(self, min_x, max_x, min_y, max_y, min_z, max_z):
        """Get random box size."""
        size = np.random.rand(3)
        size[0] = size[0] * (max_x - min_x) + min_x
        size[1] = size[1] * (max_y - min_y) + min_y
        size[2] = size[2] * (max_z - min_z) + min_z
        return size

    def color_random_brown(self, obj):
        shade = np.random.rand() + 0.5
        color = np.float32([shade * 156, shade * 117, shade * 95, 255]) / 255
        p.changeVisualShape(obj, -1, rgbaColor=color)

    def get_object_points(self, obj):
        obj_shape = p.getVisualShapeData(obj)
        obj_dim = obj_shape[0][3]
        xv, yv, zv = np.meshgrid(
            np.arange(-obj_dim[0] / 2, obj_dim[0] / 2, 0.02),
            np.arange(-obj_dim[1] / 2, obj_dim[1] / 2, 0.02),
            np.arange(-obj_dim[2] / 2, obj_dim[2] / 2, 0.02),
            sparse=False, indexing='xy')
        return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))


class PalletizingBoxes(Task):
    """Palletizing Task."""
    logger = logging.getLogger(__name__)

    def __init__(self):
        super(PalletizingBoxes, self).__init__()
        self.max_steps = 30

    def reset(self, env, random=False):
        super(PalletizingBoxes, self).reset(env)
        # Add pallet.
        zone_size = (0.3, 0.25, 0.25)
        rotation = q_convert_xyzw(q_from_euler(np.zeros(3)))
        zone_pose = np.append(np.array([0.5, 0.25, 0.02]), rotation)
        env.add_object(PALLET_URDF, zone_pose, 'fixed')
        # Add stack of boxes on pallet.
        margin = 0.01
        box_ids = []
        object_points = {}
        stack_size = (0.19, 0.19, 0.19)
        if random:
            stack_dim = np.random.randint(low=2, high=4, size=3)
        else:
            stack_dim = np.int32([2, 3, 3])
        self.box_size = (stack_size - (stack_dim - 1) * margin) / stack_dim
        box_size = copy.copy(self.box_size)
        for z in range(stack_dim[2]):
            # Transpose every layer.
            stack_dim[0], stack_dim[1] = stack_dim[1], stack_dim[0]
            box_size[0], box_size[1] = box_size[1], box_size[0]
            for y in range(stack_dim[1]):
                for x in range(stack_dim[0]):
                    position = list((x + 0.5, y + 0.5, z + 0.5) * box_size)
                    position[0] += x * margin - stack_size[0] / 2
                    position[1] += y * margin - stack_size[1] / 2
                    position[2] += z * margin + 0.03
                    pose = np.append(position, [0., 0., 0., 1.])
                    pose = p.multiplyTransforms(zone_pose[:3], zone_pose[3:], position, [0., 0., 0., 1.])
                    pose = np.append(pose[0], pose[1])
                    urdf = self.fill_template(BOX_URDF, {'DIM': box_size})
                    box_id = env.add_object(urdf, pose)
                    os.remove(urdf)
                    box_ids.append(box_id)
                    self.color_random_brown(box_id)
                    object_points[box_id] = self.get_object_points(box_id)
        # Unload consecutively top box on pallet and save ground truth pose.  # TODO: implement randomization of unload if needed
        targets = []
        self.steps = []
        boxes = copy.copy(box_ids)
        while boxes:
            box_id = boxes.pop()
            position, rotation = p.getBasePositionAndOrientation(box_id)
            rposition = np.float32(position) + np.float32([0, -10, 0])  # to hide the objects
            p.resetBasePositionAndOrientation(box_id, rposition, rotation)
            self.steps.append(box_id)
            targets.append((position, rotation))
        self.steps.reverse()  # Time-reversed depalletizing.
        self.goals.append((box_ids, targets, False, True, 'zone', (object_points, [(zone_pose, zone_size)]), 1))
        self.spawn_box()

    def reward(self):
        reward, info = super().reward()
        return reward, info

    def spawn_box(self, wait=50):
        """Palletizing: spawn another box in the workspace if it is empty."""
        workspace_empty = True
        if self.goals:
            for obj in self.goals[0][0]:
                obj_pose = p.getBasePositionAndOrientation(obj)
                workspace_empty = workspace_empty and ((obj_pose[0][1] < -0.5) or (obj_pose[0][1] > 0))
            if not self.steps:
                self.goals = []
                PalletizingBoxes.logger.warn('Palletized boxes toppled. Terminating episode.')
                return
            if workspace_empty:
                obj = self.steps[0]
                theta = np.random.random() * 2 * np.pi
                rotation = q_convert_xyzw(q_from_euler(np.array([0., 0., theta])))
                p.resetBasePositionAndOrientation(obj, [0.5, -0.25, 0.1], rotation)
                self.steps.pop(0)
        # Wait until spawned box settles.
        for _ in range(wait):
            p.stepSimulation()
