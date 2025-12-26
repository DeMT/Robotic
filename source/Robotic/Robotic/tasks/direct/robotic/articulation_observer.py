#!/usr/bin/env python3
"""
Articulation observer for monitoring robot arm and gripper state.

This module provides a composition-based observer class that wraps a
SingleArticulation to expose joint positions, velocities, end-effector pose,
and gripper state without any motion control functionality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat

from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.isaac.motion_generation")

try:
    # 新版（你原本寫的，但你這套沒有）
    from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
except ModuleNotFoundError:
    # 很多 Isaac Sim 版本實際是在這裡（OmniIsaac）
    from omni.isaac.motion_generation import LulaKinematicsSolver

from .grasp_config import PosePq


__all__ = ["ArticulationObserver", "RobotJointConfig"]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class RobotJointConfig:
    """
    Configuration for robot joint names and gripper parameters.

    Attributes:
        arm_joint_names: Names of the arm joints (in order).
        gripper_joint_names: Names of the gripper joints.
        end_effector_frame_name: Name of the end-effector frame in URDF.
        left_finger_y_offset: Left finger offset from EE along local Y-axis.
        right_finger_y_offset: Right finger offset from EE along local Y-axis.
    """

    arm_joint_names: Sequence[str] = field(default_factory=lambda: (
        "Revolute7",
        "Revolute6",
        "Revolute5",
        "Revolute4",
        "Revolute3",
        "Revolute2",
        "Revolute1",
    ))
    gripper_joint_names: Sequence[str] = field(default_factory=lambda: (
        "Slider9",
        "Slider10",
    ))
    end_effector_frame_name: str = "gripper_center"
    left_finger_y_offset: float = 0.075
    right_finger_y_offset: float = -0.075


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _resolve_path(path_str: str, label: str) -> str:
    """Resolve and validate a file path."""
    if not path_str:
        raise ValueError(f"{label} cannot be empty.")
    resolved = Path(path_str).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return str(resolved)


# ---------------------------------------------------------------------------
# ArticulationObserver
# ---------------------------------------------------------------------------

class ArticulationObserver:
    """
    Observer for monitoring robot articulation state.

    This class wraps a SingleArticulation to provide read-only access to:
    - Joint positions and velocities (full, arm subset, gripper subset)
    - End-effector pose via forward kinematics
    - Gripper finger positions and width

    No motion control or commanding functionality is included.

    Args:
        prim_path: The USD prim path of the robot articulation.
        robot_description_path: Path to the robot description YAML file for Lula.
        urdf_path: Path to the robot URDF file.
        joint_config: Robot joint configuration. If None, uses default RobotJointConfig.
        name: Optional friendly name for logging.

    Example:
        >>> observer = ArticulationObserver(
        ...     prim_path="/World/Robot",
        ...     robot_description_path="path/to/robot_description.yaml",
        ...     urdf_path="path/to/robot.urdf",
        ... )
        >>> observer.initialize()
        >>> ee_pose = observer.get_end_effector_pose()
        >>> print(f"EE position: {ee_pose.p}, orientation: {ee_pose.q}")
    """

    def __init__(
        self,
        prim_path: str,
        robot_description_path: str,
        urdf_path: str,
        joint_config: Optional[RobotJointConfig] = None,
        name: Optional[str] = None,
    ) -> None:
        self._prim_path = prim_path
        self._name = name or prim_path.rsplit("/", 1)[-1]

        # Store paths
        self._robot_description_path = robot_description_path
        self._urdf_path = urdf_path

        # Use provided config or defaults
        self._config = joint_config or RobotJointConfig()

        # Create the articulation wrapper
        self._articulation = SingleArticulation(
            prim_path=prim_path,
            name=self._name,
        )

        # Kinematics solver (initialized later)
        self._kinematics_solver: Optional[LulaKinematicsSolver] = None

        # Joint index caches (populated after initialize)
        self._arm_joint_indices: Optional[List[int]] = None
        self._gripper_joint_indices: Optional[List[int]] = None
        self._initialized = False

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The friendly name of this observer."""
        return self._name

    @property
    def prim_path(self) -> str:
        """The USD prim path of the articulation."""
        return self._prim_path

    @property
    def articulation(self) -> SingleArticulation:
        """The underlying SingleArticulation object."""
        return self._articulation

    @property
    def config(self) -> RobotJointConfig:
        """The robot joint configuration."""
        return self._config

    @property
    def num_dof(self) -> int:
        """Total number of degrees of freedom."""
        return self._articulation.num_dof

    @property
    def dof_names(self) -> List[str]:
        """List of all DOF names."""
        return list(self._articulation.dof_names)

    @property
    def arm_joint_names(self) -> List[str]:
        """Names of the arm joints."""
        return list(self._config.arm_joint_names)

    @property
    def gripper_joint_names(self) -> List[str]:
        """Names of the gripper joints."""
        return list(self._config.gripper_joint_names)

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def initialize(self, physics_sim_view=None) -> None:
        """
        Initialize the observer.

        This must be called after the simulation has started and the
        articulation is valid. It sets up:
        - The underlying articulation
        - The kinematics solver with robot base pose
        - Joint index caches for arm and gripper subsets

        Args:
            physics_sim_view: Optional physics simulation view.
        """
        if self._initialized:
            return

        # Initialize articulation
        self._articulation.initialize(physics_sim_view)

        # Build joint index caches
        dof_names = list(self._articulation.dof_names)
        print(f"[{self._name}] Available DOFs: {dof_names}")

        self._arm_joint_indices = []
        for jname in self._config.arm_joint_names:
            if jname in dof_names:
                self._arm_joint_indices.append(dof_names.index(jname))
            else:
                print(f"[{self._name}] Warning: Arm joint '{jname}' not found in articulation")

        self._gripper_joint_indices = []
        for jname in self._config.gripper_joint_names:
            if jname in dof_names:
                self._gripper_joint_indices.append(dof_names.index(jname))
            else:
                print(f"[{self._name}] Warning: Gripper joint '{jname}' not found in articulation")

        print(f"[{self._name}] Arm joint indices: {self._arm_joint_indices}")
        print(f"[{self._name}] Gripper joint indices: {self._gripper_joint_indices}")

        # Initialize kinematics solver
        robot_desc_path = _resolve_path(self._robot_description_path, "robot_description_path")
        urdf_path = _resolve_path(self._urdf_path, "urdf_path")

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=robot_desc_path,
            urdf_path=urdf_path,
        )

        # Set robot base pose for FK calculations
        position, orientation = self._articulation.get_world_pose()
        print(f"[{self._name}] Base position: {position}, orientation: {orientation}")

        def _to_cpu_numpy(x):
            import numpy as np
            import torch
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            return np.asarray(x)

        position = _to_cpu_numpy(position).reshape(3,)
        orientation = _to_cpu_numpy(orientation).reshape(4,)

        self._kinematics_solver.set_robot_base_pose(
            robot_position=position,
            robot_orientation=orientation,
        )

        self._initialized = True
        print(f"[{self._name}] ArticulationObserver initialized successfully")

    def update_robot_base_pose(self) -> None:
        """
        Update the kinematics solver with the current robot base pose.

        Call this if the robot base has moved since initialization.
        """
        if not self._initialized or self._kinematics_solver is None:
            raise RuntimeError("ArticulationObserver not initialized. Call initialize() first.")

        position, orientation = self._articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(
            robot_position=position,
            robot_orientation=orientation,
        )

    # -----------------------------------------------------------------------
    # Joint state access
    # -----------------------------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        """Get all joint positions."""
        return self._articulation.get_joint_positions()

    def get_joint_velocities(self) -> np.ndarray:
        """Get all joint velocities."""
        return self._articulation.get_joint_velocities()

    def get_arm_joint_positions(self) -> np.ndarray:
        """Get arm joint positions (7-DOF)."""
        if self._arm_joint_indices is None:
            raise RuntimeError("ArticulationObserver not initialized. Call initialize() first.")
        all_positions = self._articulation.get_joint_positions()
        # return np.array([all_positions[i] for i in self._arm_joint_indices], dtype=float)
        return (
            all_positions[self._arm_joint_indices]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )

    def get_arm_joint_velocities(self) -> np.ndarray:
        """Get arm joint velocities (7-DOF)."""
        if self._arm_joint_indices is None:
            raise RuntimeError("ArticulationObserver not initialized. Call initialize() first.")
        all_velocities = self._articulation.get_joint_velocities()
        return np.array([all_velocities[i] for i in self._arm_joint_indices], dtype=float)

    def get_gripper_joint_positions(self) -> np.ndarray:
        """
        Get gripper joint positions.

        Returns:
            Array of [Slider9, Slider10] positions.
            - Slider9: range [0, 0.05], positive = open
            - Slider10: range [-0.05, 0], negative = open
        """
        if self._gripper_joint_indices is None:
            raise RuntimeError("ArticulationObserver not initialized. Call initialize() first.")
        all_positions = self._articulation.get_joint_positions()
        return np.array([all_positions[i] for i in self._gripper_joint_indices], dtype=float)

    def get_gripper_joint_velocities(self) -> np.ndarray:
        """Get gripper joint velocities."""
        if self._gripper_joint_indices is None:
            raise RuntimeError("ArticulationObserver not initialized. Call initialize() first.")
        all_velocities = self._articulation.get_joint_velocities()
        return np.array([all_velocities[i] for i in self._gripper_joint_indices], dtype=float)

    # -----------------------------------------------------------------------
    # Forward Kinematics
    # -----------------------------------------------------------------------

    def get_end_effector_pose(
        self,
        config: Optional[np.ndarray] = None,
    ) -> PosePq:
        """
        Get the end-effector pose via forward kinematics.

        Args:
            config: Optional arm joint configuration (7-DOF). If None, uses
                current arm joint positions.

        Returns:
            PosePq with:
            - p: 3D position in world frame
            - q: quaternion in wxyz format
        """
        if self._kinematics_solver is None:
            raise RuntimeError("ArticulationObserver not initialized. Call initialize() first.")

        if config is None:
            config = self.get_arm_joint_positions()

        position, rotation = self._kinematics_solver.compute_forward_kinematics(
            frame_name=self._config.end_effector_frame_name,
            joint_positions=config,
        )
        ee_quat = rot_matrix_to_quat(rotation)
        return PosePq(
            np.asarray(position, dtype=float),
            np.asarray(ee_quat, dtype=float),
        )

    def get_finger_poses(
        self,
        config: Optional[np.ndarray] = None,
    ) -> Tuple[PosePq, PosePq]:
        """
        Get both finger poses as PosePq objects.

        Since the gripper sliders are not in the FK cspace, we compute the
        end-effector pose and apply the finger offsets along the local Y-axis.

        Args:
            config: Optional arm joint configuration (7-DOF). If None, uses
                current arm joint positions.

        Returns:
            Tuple of (left_finger_pose, right_finger_pose) in world frame.
            Each is a PosePq with:
            - p: 3D position vector
            - q: quaternion in wxyz format

        Example:
            >>> left_pose, right_pose = observer.get_finger_poses()
            >>> print(f"Left finger position: {left_pose.p}")
            >>> print(f"Left finger orientation: {left_pose.q}")
        """
        ee_pose = self.get_end_effector_pose(config)
        ee_pos = ee_pose.p
        ee_rot = quat_to_rot_matrix(ee_pose.q)

        # Get current slider positions
        gripper_pos = self.get_gripper_joint_positions()
        left_slider_pos = gripper_pos[0] if len(gripper_pos) > 0 else 0.0
        right_slider_pos = gripper_pos[1] if len(gripper_pos) > 1 else 0.0

        # Left finger offset in local frame: Y = base_offset - slider_position
        # (slider moves in -Y direction according to URDF axis)
        left_local_offset = np.array([0.0, self._config.left_finger_y_offset - left_slider_pos, 0.0])
        left_world_offset = ee_rot @ left_local_offset
        left_finger_pos = ee_pos + left_world_offset

        # Right finger offset in local frame
        right_local_offset = np.array([0.0, self._config.right_finger_y_offset - right_slider_pos, 0.0])
        right_world_offset = ee_rot @ right_local_offset
        right_finger_pos = ee_pos + right_world_offset

        # Convert rotation matrix to quaternion (wxyz format)
        ee_quat = rot_matrix_to_quat(ee_rot)

        left_pose = PosePq(
            np.asarray(left_finger_pos, dtype=float),
            np.asarray(ee_quat, dtype=float),
        )
        right_pose = PosePq(
            np.asarray(right_finger_pos, dtype=float),
            np.asarray(ee_quat, dtype=float),
        )

        return left_pose, right_pose

    # -----------------------------------------------------------------------
    # World pose access
    # -----------------------------------------------------------------------

    def get_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the robot base world pose.

        Returns:
            Tuple of (position, orientation) where orientation is quaternion (wxyz).
        """
        return self._articulation.get_world_pose().detach().cpu().numpy()
