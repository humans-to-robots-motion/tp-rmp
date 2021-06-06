from os.path import join, dirname, realpath
import numpy as np
import pybullet as p

_path_file = dirname(realpath(__file__))
ASSETS_PATH = join(_path_file, '..', '..', 'data', 'assets')
SUCTION_BASE_URDF = join(ASSETS_PATH, 'ur5', 'suction', 'suction-base.urdf')
SUCTION_HEAD_URDF = join(ASSETS_PATH, 'ur5', 'suction', 'suction-head.urdf')
GRIPPER_URDF = join(ASSETS_PATH, 'ur5', 'gripper', 'robotiq_2f_85.urdf')


class EndEffector:
    """Base ee class."""
    def __init__(self):
        self.activated = False

    def step(self):
        return

    def activate(self, objects):
        return

    def release(self):
        return


class Suction(EndEffector):
    """Simulate simple suction dynamics."""
    def __init__(self, robot_link, ee_link, obj_ids):
        super(Suction, self).__init__()
        # Load suction gripper base model (visual only).
        pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
        base = p.loadURDF(SUCTION_BASE_URDF, pose[0], pose[1])
        p.createConstraint(parentBodyUniqueId=robot_link,
                           parentLinkIndex=ee_link,
                           childBodyUniqueId=base,
                           childLinkIndex=-1,
                           jointType=p.JOINT_FIXED,
                           jointAxis=(0, 0, 0),
                           parentFramePosition=(0, 0, 0),
                           childFramePosition=(0, 0, 0.01))
        # Load suction tip model (visual and collision) with compliance.
        pose = ((0.487, 0.109, 0.347), p.getQuaternionFromEuler((np.pi, 0, 0)))
        self.suction = p.loadURDF(SUCTION_HEAD_URDF, pose[0], pose[1])
        constraint_id = p.createConstraint(parentBodyUniqueId=robot_link,
                                           parentLinkIndex=ee_link,
                                           childBodyUniqueId=self.suction,
                                           childLinkIndex=-1,
                                           jointType=p.JOINT_FIXED,
                                           jointAxis=(0, 0, 0),
                                           parentFramePosition=(0, 0, 0),
                                           childFramePosition=(0, 0, -0.08))
        p.changeConstraint(constraint_id, maxForce=50)
        # Reference to object IDs in environment for simulating suction.
        self.obj_ids = obj_ids
        # Indicates whether gripper is gripping anything (rigid or def).
        self.activated = False
        # For gripping and releasing rigid objects.
        self.contact_constraint = None

    def activate(self):
        """Simulate suction using a rigid fixed constraint to contacted object."""
        if not self.activated:
            points = p.getContactPoints(bodyA=self.suction, linkIndexA=0)
            if points:
                # Handle contact between suction with a rigid object.
                for point in points:
                    obj_id, contact_link = point[2], point[4]
                if obj_id in self.obj_ids['rigid']:
                    body_pose = p.getLinkState(self.suction, 0)
                    obj_pose = p.getBasePositionAndOrientation(obj_id)
                    world_to_body = p.invertTransform(body_pose[0],
                                                      body_pose[1])
                    obj_to_body = p.multiplyTransforms(world_to_body[0],
                                                       world_to_body[1],
                                                       obj_pose[0],
                                                       obj_pose[1])
                    self.contact_constraint = p.createConstraint(
                        parentBodyUniqueId=self.suction,
                        parentLinkIndex=0,
                        childBodyUniqueId=obj_id,
                        childLinkIndex=contact_link,
                        jointType=p.JOINT_FIXED,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=obj_to_body[0],
                        parentFrameOrientation=obj_to_body[1],
                        childFramePosition=(0, 0, 0),
                        childFrameOrientation=(0, 0, 0))
                self.activated = True

    def release(self):
        """Release gripper object, only applied if gripper is 'activated'."""
        if self.activated:
            self.activated = False
            # Release gripped rigid object (if any).
            if self.contact_constraint is not None:
                try:
                    p.removeConstraint(self.contact_constraint)
                    self.contact_constraint = None
                except:  # noqa
                    pass

    def detect_contact(self):
        """Detects a contact with a rigid object."""
        body, link = self.suction, 0
        if self.activated and self.contact_constraint is not None:
            try:
                info = p.getConstraintInfo(self.contact_constraint)
                body, link = info[2], info[3]
            except:  # noqa
                self.contact_constraint = None
                pass
        # Get all contact points between the suction and a rigid body.
        points = p.getContactPoints(bodyA=body, linkIndexA=link)
        if self.activated:
            points = [point for point in points if point[2] != self.suction]
        # # We know if len(points) > 0, contact is made with SOME rigid item.
        if points:
            return True
        return False

    def check_grasp(self):
        """Check a grasp (object in contact?) for picking success."""
        suctioned_object = None
        if self.contact_constraint is not None:
            suctioned_object = p.getConstraintInfo(self.contact_constraint)[2]
        return suctioned_object is not None


class Gripper(EndEffector):
    """Simulate simple gripper."""
    def __init__(self, robot_link, ee_link, obj_ids):
        """Creates gripper and attaches it to the robot."""
        super(Gripper, self).__init__()
        self.robot_link = robot_link
        self.ee_link = ee_link
        self.obj_ids = obj_ids
        self.contact_constraint = None
        pose = ((0.487, 0.109, 0.347), p.getQuaternionFromEuler((np.pi, 0, 0)))
        self.gripper_link = p.loadURDF(GRIPPER_URDF, pose[0], pose[1])
        constraint_id = p.createConstraint(  # noqa
            parentBodyUniqueId=robot_link,
            parentLinkIndex=ee_link,
            childBodyUniqueId=self.gripper_link,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.08))
        # p.changeConstraint(constraint_id, maxForce=50)

    def activate(self):
        """Simulate suction using a rigid fixed constraint to contacted object."""
        # TODO: implement gripper logic
        pass
