import copy
from typing import Callable, List
from pathlib import Path

import rospy
import actionlib
import numpy as np
import tf
from threading import Lock
from typing import List, Union
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryGoal
from controller_manager_msgs.srv import SwitchController, LoadController, ListControllers
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from franka_example_controllers.msg import JointStateFiltered

class ControllerManager:

    def __init__(self, desired_controllers: List[str]):
        rospy.wait_for_service("/controller_manager/list_controllers")
        rospy.wait_for_service("/controller_manager/load_controller")
        rospy.wait_for_service("/controller_manager/switch_controller")

        load_controller_service = rospy.ServiceProxy("/controller_manager/load_controller", LoadController)
        list_controller_service = rospy.ServiceProxy("/controller_manager/list_controllers", ListControllers)
        self.switch_controller_service = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)

        res = list_controller_service()
        controller_names = [r.name for r in res.controller]

        for desired_controller in desired_controllers:
            if desired_controller not in controller_names:
                res = load_controller_service(desired_controller)
                if not res:
                    raise RuntimeError(f"Could not load {desired_controller}")

    def activate(self: str, new_controller, stop_controllers: List[str] = None):
        if stop_controllers is None:
            stop_controllers = []

        res = self.switch_controller_service([new_controller], stop_controllers, 1, False, 10.)
        if not res.ok:
            raise RuntimeError(f"Could not start controller {new_controller}")
        
class SafetyModule:

    def __init__(self, transform_listener: tf.TransformListener, ref_frame_id: str, frame_id: str, pos_lb: np.ndarray,
                 pos_ub: np.ndarray, stop_callback: Callable):
        self.transform_listener = transform_listener
        self.stop_callback = stop_callback
        self.stopped = False
        self.ref_frame_id = ref_frame_id
        self.frame_id = frame_id
        self.pos_lb = pos_lb
        self.pos_ub = pos_ub

    def stop(self):
        self.stopped = True

    def monitor(self):
        while not self.stopped:
            frame_pos = np.array(self.transform_listener.lookupTransform(self.ref_frame_id,
                                                                         self.frame_id, rospy.Time(0.))[0])
            if np.any(frame_pos < self.pos_lb) or np.any(self.pos_ub < frame_pos):
                self.stop_callback()

            rospy.sleep(rospy.Duration(0.01))


class StopHelper:

    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class Limits:

    def __init__(self):
        # qpos limits
        self.joint_qpos_shift = np.array([0, 0, 0, 0, 0, -np.pi / 2, -np.pi / 4])
        self.joint_qpos_min = np.array([-166, -101, -166, -176, -166, -1, -166])
        self.joint_qpos_min = self.joint_qpos_min / 180 * np.pi + self.joint_qpos_shift
        self.joint_qpos_max = np.array([166, 101, 166, -4, 166, 215, 166])
        self.joint_qpos_max = self.joint_qpos_max / 180 * np.pi + self.joint_qpos_shift

        # qvel limits
        joint_qvel_limits = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
        self.joint_qvel_min = -joint_qvel_limits
        self.joint_qvel_max = joint_qvel_limits

        # qacc limits (TODO maybe we want to rescale them to the original values)
        self.joint_qacc_max = 0.2 * np.array([15, 7.5, 10, 12.5, 15, 20, 20])


def go_to(trajectory_client: actionlib.SimpleActionClient, joint_names: List[str],
          init_positions: Union[np.ndarray, List[float]], target_positions: Union[np.ndarray, List[float]],
          duration: float = None, max_vel: float = None, wait: bool = True):
    if duration is None:
        if max_vel is None:
            raise RuntimeError("If duration is not specified, the maximum allowed joint velocity must be specified")

        duration = max(0.5, np.linalg.norm(np.array(init_positions) - np.array(target_positions)) / max_vel)

    # Create a trajectory in which we simply move J0 by 20 degrees
    traj = JointTrajectory()
    traj.joint_names = joint_names

    traj_point = JointTrajectoryPoint()
    traj_point.time_from_start = rospy.Duration(duration)
    traj_point.positions = target_positions
    traj_point.velocities = [0] * len(target_positions)
    traj.points = [traj_point]

    traj_goal = FollowJointTrajectoryGoal()
    traj_goal.trajectory = traj
    trajectory_client.send_goal(traj_goal)
    if wait:
        trajectory_client.wait_for_result(rospy.Duration.from_sec(duration))
        return trajectory_client.get_result()


class JointStateListener:

    def __init__(self):
        self.sub = rospy.Subscriber("/joint_states", JointState, self._ros_callback)
        self.last_message = None
        self.lock = Lock()

    def _ros_callback(self, data: JointState):
        self.lock.acquire()
        self.last_message = data
        self.lock.release()

    def get_data(self) -> JointState:
        self.lock.acquire()
        data = copy.deepcopy(self.last_message)
        self.lock.release()
        return data

class JointStateFilteredListener:

    def __init__(self, inv_dyn_function):
        self.sub = rospy.Subscriber("/feedforward_joint_pd_controller/filtered_joint_states", JointStateFiltered, self._ros_callback)
        self.last_message = None
        self.inv_dyn_function = inv_dyn_function
        self.tau_cutoff_frequency = 15
        filter_Ts = 1.0 / 500.0
        self.tau_alpha = filter_Ts / (filter_Ts + 1.0 / (2 * np.pi * self.tau_cutoff_frequency))
        self.old_tau_nom_filtered = np.zeros(7)
        self.old_diff_tau_nom = np.zeros(7)
        self.lock = Lock()

    def _ros_callback(self, data: JointStateFiltered):
        self.lock.acquire()
        self.last_message = data
        tau_nom = np.array(self.inv_dyn_function(data.position, data.velocity, data.acceleration)).squeeze()
        tau_nom_filtered = first_order_low_pass_filter(tau_nom, self.old_tau_nom_filtered, self.tau_alpha)
        self.old_tau_nom_filtered = np.copy(tau_nom_filtered)

        self.last_message.tau_nom = tau_nom.tolist()
        self.last_message.tau_nom_filtered = tau_nom_filtered.tolist()
        self.last_message.diff_tau_nom = self.last_message.effort - tau_nom
        self.lock.release()

    def get_data(self) -> JointStateFiltered:
        self.lock.acquire()
        data = copy.deepcopy(self.last_message)
        self.lock.release()
        return data


def first_order_low_pass_filter(value, old_value, alpha):
    return alpha * value + (1 - alpha) * old_value
