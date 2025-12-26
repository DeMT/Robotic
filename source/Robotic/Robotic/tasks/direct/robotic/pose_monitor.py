#!/usr/bin/env python3
"""
Pose monitor for tracking spatial relationships between robot and target objects.

This module provides a composition-based monitor class that tracks pose errors
between the robot's end-effector and target objects (fan, ground truth position).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from .articulation_observer import ArticulationObserver, RobotJointConfig
from .grasp_config import ApproachFrameConfig, GraspDetectionConfig, PosePq
from .target_object import TargetObject
from isaacsim.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat


__all__ = [
    "PoseError",
    "ApproachFrameConfig",
    "GraspDetectionConfig",
    "GraspDetectionStrategy",
    "PoseMonitor",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PoseError:
    """
    Pose error between two frames.

    Attributes:
        position_error: Position difference vector (3D) from source to target.
        distance: Euclidean distance in meters.
        rotation_error: Relative rotation matrix (3x3) from source to target.
        angle_error: Rotation angle error in radians (magnitude of rotation).
    """

    position_error: np.ndarray
    distance: float
    rotation_error: np.ndarray
    angle_error: float

    def __repr__(self) -> str:
        return (
            f"PoseError(distance={self.distance:.4f}m, "
            f"angle_error={np.degrees(self.angle_error):.2f}°)"
        )


GraspState = Literal["open", "closed", "holding"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _to_pose_pq(position: np.ndarray, rotation_or_quat: np.ndarray) -> PosePq:
    pos = np.asarray(position, dtype=float).reshape(3)
    arr = np.asarray(rotation_or_quat, dtype=float)
    if arr.shape == (3, 3):
        quat = rot_matrix_to_quat(arr)
    else:
        quat = arr.reshape(4)
    return PosePq(pos, np.asarray(quat, dtype=float))


def _compute_pose_error_pq(source: PosePq, target: PosePq) -> PoseError:
    position_error = np.asarray(target.p, dtype=float) - np.asarray(source.p, dtype=float)
    distance = float(np.linalg.norm(position_error))

    src_rot = quat_to_rot_matrix(source.q)
    tgt_rot = quat_to_rot_matrix(target.q)
    rotation_error = tgt_rot @ src_rot.T
    rot_quat = rot_matrix_to_quat(rotation_error)
    rot_quat = np.asarray(rot_quat, dtype=float)
    angle_error = float(2.0 * np.arccos(np.clip(rot_quat[0] / np.linalg.norm(rot_quat), -1.0, 1.0)))

    return PoseError(
        position_error=position_error,
        distance=distance,
        rotation_error=rotation_error,
        angle_error=angle_error,
    )


# ---------------------------------------------------------------------------
# GraspDetectionStrategy
# ---------------------------------------------------------------------------

class GraspDetectionStrategy:
    """
    Strategy for detecting whether the gripper is holding an object.

    This class evaluates gripper state based on instantaneous measurements:
    - Gripper joint positions within symmetric threshold range
    - End-effector distance to target (whether within grasp zone)

    Args:
        config: Grasp detection configuration. If None, uses default values.

    Example:
        >>> # Use default config
        >>> strategy = GraspDetectionStrategy()
        >>> 
        >>> # Use custom config
        >>> config = GraspDetectionConfig(
        ...     grip_position_min=0.018,
        ...     grip_position_max=0.022,
        ...     grasp_zone_max_m=0.03,
        ... )
        >>> strategy = GraspDetectionStrategy(config)
        >>> 
        >>> is_holding = strategy.evaluate(
        ...     gripper_positions=np.array([0.02, -0.02]),
        ...     ee_distance_m=0.03,
        ... )
    """

    def __init__(self, config: Optional[GraspDetectionConfig] = None) -> None:
        # Handle None to avoid mutable default argument issue
        self._config = config if config is not None else GraspDetectionConfig()

        # Internal state (for debug)
        self._state: GraspState = "open"

    # -----------------------------------------------------------------------
    # Properties (for debug)
    # -----------------------------------------------------------------------

    @property
    def config(self) -> GraspDetectionConfig:
        """The grasp detection configuration."""
        return self._config

    @property
    def state(self) -> GraspState:
        """Current grasp state (for debug purposes)."""
        return self._state

    # -----------------------------------------------------------------------
    # Core methods
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        gripper_positions: np.ndarray,
        ee_distance_m: float,
    ) -> bool:
        """
        Evaluate whether the gripper is currently holding an object.

        This is an instantaneous check based on current gripper state
        and distance to target.

        Grip detection logic (based on empirical testing):
        - Slider9 (left finger): ~+0.02m when holding
        - Slider10 (right finger): ~-0.02m when holding
        - Uses symmetric range for both joints:
          - Slider9:  grip_position_min <= value <= grip_position_max
          - Slider10: -grip_position_max <= value <= -grip_position_min

        Args:
            gripper_positions: Array of gripper joint positions [Slider9, Slider10].
            ee_distance_m: Distance from end-effector to target in meters.

        Returns:
            True if currently holding (gripper closed and within grasp zone),
            False otherwise.
        """
        cfg = self._config

        # Symmetric range grip detection
        slider9 = float(gripper_positions[0])
        slider10 = float(gripper_positions[1])
        
        slider9_ok = cfg.grip_position_min <= slider9 <= cfg.grip_position_max
        slider10_ok = -cfg.grip_position_max <= slider10 <= -cfg.grip_position_min
        is_closed = slider9_ok and slider10_ok

        # Check if within grasp zone
        is_in_grasp_zone = cfg.grasp_zone_min_m <= ee_distance_m <= cfg.grasp_zone_max_m

        # Update internal state
        if not is_closed:
            self._state = "open"
        elif is_closed and is_in_grasp_zone:
            self._state = "holding"
        else:
            self._state = "closed"

        return self._state == "holding"


# ---------------------------------------------------------------------------
# PoseMonitor
# ---------------------------------------------------------------------------

class PoseMonitor:
    """
    Monitor for tracking pose errors between robot end-effector and target objects.

    This class uses the Strategy/Composition pattern to combine an ArticulationObserver
    (for robot state) with TargetObject instances (for target poses) to compute
    spatial relationships.

    Args:
        robot_observer: The ArticulationObserver for the robot.
        fan_object: Optional TargetObject for the fan.
        ground_truth_object: Optional TargetObject for the ground truth pose.

    Example:
        >>> observer = ArticulationObserver(...)
        >>> observer.initialize()
        >>> fan = TargetObject("/World/Fan")
        >>> ground_truth = TargetObject("/World/GroundTruth")
        >>> monitor = PoseMonitor(
        ...     robot_observer=observer,
        ...     fan_object=fan,
        ...     ground_truth_object=ground_truth,
        ... )
        >>> ee_to_fan = monitor.get_ee_to_fan_error()
        >>> print(f"Distance to fan: {ee_to_fan.distance:.3f}m")
        >>> print(f"Angle error: {np.degrees(ee_to_fan.angle_error):.1f}°")
    """

    def __init__(
        self,
        robot_observer: ArticulationObserver,
        fan_object: Optional[TargetObject] = None,
        ground_truth_object: Optional[TargetObject] = None,
        grasp_strategy: Optional[GraspDetectionStrategy] = None,
    ) -> None:
        self.robot_observer = robot_observer
        self.fan_object = fan_object
        self.ground_truth_object = ground_truth_object
        self.grasp_strategy = grasp_strategy

    # -----------------------------------------------------------------------
    # End-effector pose access
    # -----------------------------------------------------------------------

    def get_end_effector_pose(self) -> PosePq:
        """
        Get the current end-effector pose.

        Returns:
            PosePq with:
            - p: 3D position in world frame
            - q: quaternion in wxyz format
        """
        return self.robot_observer.get_end_effector_pose()

    # -----------------------------------------------------------------------
    # Joint position access
    # -----------------------------------------------------------------------

    def get_arm_joint_positions(self) -> np.ndarray:
        """
        Get the current arm joint positions.

        Returns:
            Array of arm joint positions (7-DOF).
        """
        return self.robot_observer.get_arm_joint_positions().detach().cpu().numpy()

    def get_gripper_joint_positions(self) -> np.ndarray:
        """
        Get the current gripper joint positions.

        Returns:
            Array of gripper joint positions [Slider9, Slider10].
        """
        return self.robot_observer.get_gripper_joint_positions()

    # -----------------------------------------------------------------------
    # Pose error computation
    # -----------------------------------------------------------------------

    def get_ee_to_fan_error(self) -> PoseError:
        """
        Compute pose error between end-effector and fan.

        Returns:
            PoseError containing position and rotation errors.

        Raises:
            ValueError: If fan_object is not set.
        """
        if self.fan_object is None:
            raise ValueError("fan_object is not set.")

        # Get EE pose
        ee_pose = self.robot_observer.get_end_effector_pose()

        # Get fan pose
        fan_pos, fan_quat = self.fan_object.get_world_pose()
        fan_pose = _to_pose_pq(fan_pos, fan_quat)

        return _compute_pose_error_pq(ee_pose, fan_pose)

    def get_ee_to_ground_truth_error(self) -> PoseError:
        """
        Compute pose error between end-effector and ground truth position.

        Returns:
            PoseError containing position and rotation errors.

        Raises:
            ValueError: If ground_truth_object is not set.
        """
        if self.ground_truth_object is None:
            raise ValueError("ground_truth_object is not set.")

        # Get EE pose
        ee_pose = self.robot_observer.get_end_effector_pose()

        # Get ground truth pose
        gt_pos, gt_quat = self.ground_truth_object.get_world_pose()
        gt_pose = _to_pose_pq(gt_pos, gt_quat)

        return _compute_pose_error_pq(ee_pose, gt_pose)

    def get_pose_error_to_target(self, target: TargetObject) -> PoseError:
        """
        Compute pose error between end-effector and an arbitrary target.

        This is a general method that can be used with any TargetObject.

        Args:
            target: The target object to compute error against.

        Returns:
            PoseError containing position and rotation errors.
        """
        # Get EE pose
        ee_pose = self.robot_observer.get_end_effector_pose()

        # Get target pose
        target_pos, target_quat = target.get_world_pose()
        target_pose = _to_pose_pq(target_pos, target_quat)

        return _compute_pose_error_pq(ee_pose, target_pose)

    # -----------------------------------------------------------------------
    # Grasp detection
    # -----------------------------------------------------------------------

    def is_holding_fan(self) -> bool:
        """
        Check if the gripper is currently holding the fan.

        This method uses the configured grasp strategy to evaluate whether
        the gripper is in a holding state based on gripper positions and
        distance to the fan.

        Returns:
            True if holding the fan, False otherwise.

        Raises:
            ValueError: If grasp_strategy or fan_object is not set.
        """
        if self.grasp_strategy is None:
            raise ValueError("grasp_strategy is not set.")
        if self.fan_object is None:
            raise ValueError("fan_object is not set.")

        # Get gripper positions
        gripper_positions = self.robot_observer.get_gripper_joint_positions()

        # Get EE distance to fan
        ee_to_fan = self.get_ee_to_fan_error()
        ee_distance = ee_to_fan.distance

        return self.grasp_strategy.evaluate(
            gripper_positions=gripper_positions,
            ee_distance_m=ee_distance,
        )

    # -----------------------------------------------------------------------
    # Finger to handle distance
    # -----------------------------------------------------------------------

    def get_finger_to_handle_distances(self) -> Tuple[float, float]:
        """
        Get distances from gripper fingers to fan handles.

        This provides more precise grasp quality information than
        EE-to-fan-center distance, useful for RL reward shaping or
        grasp quality constraints.

        The handle positions are computed from the fan center using
        offsets from the fan_object's grasp_config.

        Returns:
            Tuple of (left_finger_to_left_handle_dist, right_finger_to_right_handle_dist)
            in meters.

        Raises:
            ValueError: If fan_object is not set.
            ValueError: If fan_object's grasp_config is not set.

        Example:
            >>> left_dist, right_dist = monitor.get_finger_to_handle_distances()
            >>> print(f"Left: {left_dist:.3f}m, Right: {right_dist:.3f}m")
        """
        if self.fan_object is None:
            raise ValueError("fan_object is not set.")

        # Get finger poses from robot observer
        left_finger_pose, right_finger_pose = self.robot_observer.get_finger_poses()

        # Get handle poses from fan object (uses its grasp_config)
        left_handle_pose, right_handle_pose = self.fan_object.get_handle_poses()

        # Compute distances using position component (.p) of PosePq
        left_dist = float(np.linalg.norm(left_finger_pose.p - left_handle_pose.p))
        right_dist = float(np.linalg.norm(right_finger_pose.p - right_handle_pose.p))

        return left_dist, right_dist

    def get_finger_poses(self) -> Tuple[PosePq, PosePq]:
        """
        Get the current gripper finger poses in world frame.

        This is a convenience method that delegates to the robot observer.

        Returns:
            Tuple of (left_finger_pose, right_finger_pose) in world frame.
            Each is a PosePq with:
            - p: 3D position vector
            - q: quaternion in wxyz format
        """
        return self.robot_observer.get_finger_poses()

    def get_handle_poses(self) -> Tuple[PosePq, PosePq]:
        """
        Get the virtual fan handle poses in world frame.

        This is a convenience method that delegates to the fan object.

        Returns:
            Tuple of (left_handle_pose, right_handle_pose) in world frame.
            Each is a PosePq with:
            - p: 3D position vector
            - q: quaternion in wxyz format

        Raises:
            ValueError: If fan_object is not set.
            ValueError: If fan_object's grasp_config is not set.
        """
        if self.fan_object is None:
            raise ValueError("fan_object is not set.")

        return self.fan_object.get_handle_poses()

    # -----------------------------------------------------------------------
    # Factory method
    # -----------------------------------------------------------------------

    @classmethod
    def create_default(
        cls,
        robot_prim_path: str,
        fan_prim_path: str,
        ground_truth_prim_path: str,
        robot_description_path: str,
        urdf_path: str,
        joint_config: Optional[RobotJointConfig] = None,
        grasp_config: Optional[GraspDetectionConfig] = None,
    ) -> "PoseMonitor":
        """
        Factory method to create a PoseMonitor.

        This method simplifies initialization by automatically creating the
        ArticulationObserver and TargetObject instances from prim paths.

        All prim paths and robot configuration paths are REQUIRED and must 
        point to existing resources.

        Args:
            robot_prim_path: USD prim path of the robot articulation. REQUIRED.
            fan_prim_path: USD prim path of the fan object. REQUIRED.
            ground_truth_prim_path: USD prim path of the ground truth object. REQUIRED.
            robot_description_path: Path to robot description YAML. REQUIRED.
            urdf_path: Path to robot URDF. REQUIRED.
            joint_config: Robot joint configuration. Defaults to RobotJointConfig().
            grasp_config: Grasp detection configuration including target frame alignment.
                          If None, uses default GraspDetectionConfig (no frame transformation).

        Returns:
            Configured PoseMonitor instance (call initialize() before use).

        Raises:
            ValueError: If any of the required parameters is empty or None.

        Example:
            >>> from grasp_config import GraspDetectionConfig
            >>> # Create config for fan (front=-Y, grasp axis=+X)
            >>> config = GraspDetectionConfig()
            >>> config.approach_axis = "-y"
            >>> config.grasp_axis = "+x"
            >>> monitor = PoseMonitor.create_default(
            ...     robot_prim_path="/World/WorkSpace/RS_M90E7A_Left",
            ...     fan_prim_path="/World/WorkSpace/Scene/Fan",
            ...     ground_truth_prim_path="/World/WorkSpace/Scene/GroundTruth",
            ...     robot_description_path="path/to/robot_description.yaml",
            ...     urdf_path="path/to/robot.urdf",
            ...     grasp_config=config,
            ... )
            >>> monitor.initialize()
            >>> error = monitor.get_ee_to_fan_error()
        """
        # Validate required parameters
        errors = []
        if not robot_prim_path or not robot_prim_path.strip():
            errors.append("robot_prim_path is required but was empty or None")
        if not fan_prim_path or not fan_prim_path.strip():
            errors.append("fan_prim_path is required but was empty or None")
        if not ground_truth_prim_path or not ground_truth_prim_path.strip():
            errors.append("ground_truth_prim_path is required but was empty or None")
        if not robot_description_path or not robot_description_path.strip():
            errors.append("robot_description_path is required but was empty or None")
        if not urdf_path or not urdf_path.strip():
            errors.append("urdf_path is required but was empty or None")
        
        if errors:
            raise ValueError(
                "PoseMonitor.create_default() requires all parameters:\n  - " 
                + "\n  - ".join(errors)
            )

        # Use default config if not provided
        if grasp_config is None:
            grasp_config = GraspDetectionConfig()

        # Create ArticulationObserver
        robot_observer = ArticulationObserver(
            prim_path=robot_prim_path,
            robot_description_path=robot_description_path,
            urdf_path=urdf_path,
            joint_config=joint_config,
        )

        # Create TargetObjects with grasp config (contains target_frame + handle offsets)
        fan_object = TargetObject(
            prim_path=fan_prim_path,
            grasp_config=grasp_config,
        )
        ground_truth_object = TargetObject(
            prim_path=ground_truth_prim_path,
            grasp_config=grasp_config,
        )

        # Create GraspDetectionStrategy with the config
        grasp_strategy = GraspDetectionStrategy(grasp_config)

        return cls(
            robot_observer=robot_observer,
            fan_object=fan_object,
            ground_truth_object=ground_truth_object,
            grasp_strategy=grasp_strategy,
        )

    def initialize(self, physics_sim_view=None) -> None:
        """
        Initialize the pose monitor.

        This initializes the underlying ArticulationObserver. Must be called
        after the simulation has started and the articulation is valid.

        Args:
            physics_sim_view: Optional physics simulation view.
        """
        self.robot_observer.initialize(physics_sim_view)


