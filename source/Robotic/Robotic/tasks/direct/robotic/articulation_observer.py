#!/usr/bin/env python3
"""
Articulation observer for monitoring robot arm and gripper state.

This module provides a composition-based observer class that wraps a
SingleArticulation to expose joint positions, velocities, end-effector pose,
and gripper state without any motion control functionality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import isaacsim.core.utils.stage as stage_utils
import omni.physics.tensors.impl.api as physx
from pxr import UsdPhysics
from isaacsim.core.prims import SingleArticulation

from .array_backend import mathops as mo
from .grasp_config import PosePq


__all__ = ["ArticulationObserver", "RobotJointConfig"]

try:
    from isaacsim.core.api.simulation_context.simulation_context import SimulationContext
except Exception:  # pragma: no cover
    SimulationContext = None


def _quat_xyzw_to_wxyz(quat):
    """Convert quaternion from xyzw to wxyz (supports torch tensors and numpy arrays)."""
    if isinstance(quat, torch.Tensor):
        return quat[..., (3, 0, 1, 2)]
    quat_arr = np.asarray(quat)
    return quat_arr[..., (3, 0, 1, 2)]


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
        end_effector_tf1_prim_suffix: TF_1 prim path suffix from robot root.
        tf1_to_grasp_center_offset: Fixed offset from TF_1 to grasp center in TF_1 local frame.
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
    end_effector_tf1_prim_suffix: str = "TF_1"
    tf1_to_grasp_center_offset: Sequence[float] = field(
        default_factory=lambda: (0.118, 0.0, -0.003)
    )
    left_finger_y_offset: float = 0.075
    right_finger_y_offset: float = -0.075


# ---------------------------------------------------------------------------
# ArticulationObserver
# ---------------------------------------------------------------------------

class ArticulationObserver:
    """
    Observer for monitoring robot articulation state.

    This class wraps a SingleArticulation to provide read-only access to:
    - Joint positions and velocities (full, arm subset, gripper subset)
    - End-effector pose via TF_1 with a fixed grasp center offset
    - Gripper finger positions and width

    No motion control or commanding functionality is included.

    Args:
        prim_path: The USD prim path of the robot articulation.
        joint_config: Robot joint configuration. If None, uses default RobotJointConfig.
        name: Optional friendly name for logging.

    Example:
        >>> observer = ArticulationObserver(
        ...     prim_path="/World/Robot",
        ... )
        >>> observer.initialize()
        >>> ee_pose = observer.get_end_effector_pose()
        >>> print(f"EE position: {ee_pose.p}, orientation: {ee_pose.q}")
    """

    def __init__(
        self,
        prim_path: str,
        joint_config: Optional[RobotJointConfig] = None,
        name: Optional[str] = None,
    ) -> None:
        self._prim_path = prim_path
        self._name = name or prim_path.rsplit("/", 1)[-1]

        # Use provided config or defaults
        self._config = joint_config or RobotJointConfig()

        # Create the articulation wrapper
        self._articulation = SingleArticulation(
            prim_path=prim_path,
            name=self._name,
        )

        # PhysX tensor views (initialized later)
        self._physics_sim_view = None
        self._root_physx_view = None
        self._tf1_link_index: Optional[int] = None

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
        return int(self._root_physx_view.shared_metatype.dof_count)

    @property
    def dof_names(self) -> List[str]:
        """List of all DOF names."""
        return list(self._root_physx_view.shared_metatype.dof_names)

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
        - The end-effector TF_1 prim wrapper
        - Joint index caches for arm and gripper subsets

        Args:
            physics_sim_view: Optional physics simulation view.
        """
        if self._initialized:
            return

        # Initialize articulation
        self._articulation.initialize(physics_sim_view)

        # Create PhysX tensor articulation view (hard-coded root prim suffix)
        root_joint_path = f"{self._prim_path.rstrip('/')}/root_joint"
        stage = stage_utils.get_current_stage()
        prim = stage.GetPrimAtPath(root_joint_path)
        if not prim.IsValid():
            raise RuntimeError(
                f"[{self._name}] Invalid articulation root prim: '{root_joint_path}'. "
                "Expected robot prim path to have a '/root_joint' child."
            )
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise RuntimeError(
                f"[{self._name}] Prim '{root_joint_path}' is not an articulation root "
                "(missing UsdPhysics.ArticulationRootAPI)."
            )

        if (
            physics_sim_view is not None
            and hasattr(physics_sim_view, "create_articulation_view")
            and hasattr(physics_sim_view, "update_articulations_kinematic")
        ):
            self._physics_sim_view = physics_sim_view
        else:
            backend = "torch"
            if SimulationContext is not None:
                ctx = SimulationContext.instance()
                ctx_backend = getattr(ctx, "backend", None)
                if ctx_backend is not None:
                    backend = str(ctx_backend).strip().lower()
            if backend not in ("torch", "numpy"):
                backend = "torch"
            self._physics_sim_view = physx.create_simulation_view(backend)
            self._physics_sim_view.set_subspace_roots("/")

        root_joint_expr = root_joint_path.replace(".*", "*")
        self._root_physx_view = self._physics_sim_view.create_articulation_view(root_joint_expr)
        if getattr(self._root_physx_view, "_backend", None) is None:
            raise RuntimeError(f"[{self._name}] Failed to create PhysX articulation view for '{root_joint_expr}'.")

        # Cache TF_1 link index from PhysX metadata
        tf1_name = str(self._config.end_effector_tf1_prim_suffix).strip("/").split("/")[-1]
        link_names = list(self._root_physx_view.shared_metatype.link_names)
        if tf1_name not in link_names:
            raise RuntimeError(f"[{self._name}] Link '{tf1_name}' not found. Available links: {link_names}")
        self._tf1_link_index = link_names.index(tf1_name)

        # Build joint index caches
        dof_names = list(self._root_physx_view.shared_metatype.dof_names)
        print(f"[{self._name}] Available DOFs: {dof_names}")

        self._arm_joint_indices = []
        for jname in self._config.arm_joint_names:
            if jname not in dof_names:
                raise RuntimeError(
                    f"[{self._name}] Arm joint '{jname}' not found in articulation DOFs: {dof_names}"
                )
            self._arm_joint_indices.append(dof_names.index(jname))

        self._gripper_joint_indices = []
        for jname in self._config.gripper_joint_names:
            if jname not in dof_names:
                raise RuntimeError(
                    f"[{self._name}] Gripper joint '{jname}' not found in articulation DOFs: {dof_names}"
                )
            self._gripper_joint_indices.append(dof_names.index(jname))


        self._initialized = True
        print(f"[{self._name}] ArticulationObserver initialized successfully")


    # -----------------------------------------------------------------------
    # Joint state access
    # -----------------------------------------------------------------------

    def get_joint_positions(self) -> Any:
        """Get all joint positions."""
        self._physics_sim_view.update_articulations_kinematic()
        positions = self._root_physx_view.get_dof_positions()
        return positions[0]

    def get_joint_velocities(self) -> Any:
        """Get all joint velocities."""
        self._physics_sim_view.update_articulations_kinematic()
        velocities = self._root_physx_view.get_dof_velocities()
        return velocities[0]

    def get_arm_joint_positions(self) -> Any:
        """Get arm joint positions (7-DOF)."""
        all_positions = self.get_joint_positions()
        return all_positions[self._arm_joint_indices]

    def get_arm_joint_velocities(self) -> Any:
        """Get arm joint velocities (7-DOF)."""
        all_velocities = self.get_joint_velocities()
        return all_velocities[self._arm_joint_indices]

    def get_gripper_joint_positions(self) -> Any:
        """
        Get gripper joint positions.

        Returns:
            Array of [Slider9, Slider10] positions.
            - Slider9: range [0, 0.05], positive = open
            - Slider10: range [-0.05, 0], negative = open
        """
        all_positions = self.get_joint_positions()
        return all_positions[self._gripper_joint_indices]

    def get_gripper_joint_velocities(self) -> Any:
        """Get gripper joint velocities."""
        all_velocities = self.get_joint_velocities()
        return all_velocities[self._gripper_joint_indices]

    # -----------------------------------------------------------------------
    # End-effector pose
    # -----------------------------------------------------------------------

    def get_end_effector_pose(
        self,
        config: Optional[Any] = None,
    ) -> PosePq:
        """
        Get the end-effector pose from TF_1 with a fixed grasp center offset.

        Args:
            config: Unused. Present for API compatibility.

        Returns:
            PosePq with:
            - p: 3D position in world frame
            - q: quaternion in wxyz format
        """
        self._physics_sim_view.update_articulations_kinematic()
        poses = self._root_physx_view.get_link_transforms()
        tf1_pose = poses[0, self._tf1_link_index]
        tf1_pos = tf1_pose[:3]
        tf1_quat = _quat_xyzw_to_wxyz(tf1_pose[3:7])
        tf1_rot = mo.quat_to_rot_matrix(tf1_quat)

        if isinstance(tf1_pos, torch.Tensor):
            ee_offset = torch.as_tensor(
                self._config.tf1_to_grasp_center_offset,
                device=tf1_pos.device,
                dtype=tf1_pos.dtype,
            )
        else:
            ee_offset = np.asarray(self._config.tf1_to_grasp_center_offset, dtype=np.float32)
        ee_pos = tf1_pos + tf1_rot @ ee_offset
        return PosePq(ee_pos, tf1_quat)

    def get_finger_poses(
        self,
        config: Optional[Any] = None,
    ) -> Tuple[PosePq, PosePq]:
        """
        Get both finger poses as PosePq objects.

        Since the gripper sliders are not part of the end-effector pose, we
        compute the end-effector pose and apply the finger offsets along
        the local Y-axis.

        Args:
            config: Unused. Present for API compatibility.

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
        ee_rot = mo.quat_to_rot_matrix(ee_pose.q)

        # Get current slider positions
        gripper_pos = self.get_gripper_joint_positions()
        left_slider_pos = gripper_pos[0] if len(gripper_pos) > 0 else 0.0
        right_slider_pos = gripper_pos[1] if len(gripper_pos) > 1 else 0.0

        # Left finger offset in local frame: Y = base_offset - slider_position
        # (slider moves in -Y direction according to URDF axis)
        y_axis = mo.asarray([0.0, 1.0, 0.0])
        left_local_offset = (
            mo.asarray([0.0, self._config.left_finger_y_offset, 0.0])
            - y_axis * left_slider_pos
        )
        left_world_offset = ee_rot @ left_local_offset
        left_finger_pos = ee_pos + left_world_offset

        # Right finger offset in local frame
        right_local_offset = (
            mo.asarray([0.0, self._config.right_finger_y_offset, 0.0])
            - y_axis * right_slider_pos
        )
        right_world_offset = ee_rot @ right_local_offset
        right_finger_pos = ee_pos + right_world_offset

        # Convert rotation matrix to quaternion (wxyz format)
        ee_quat = mo.rot_matrix_to_quat(ee_rot)

        left_pose = PosePq(left_finger_pos, ee_quat)
        right_pose = PosePq(right_finger_pos, ee_quat)

        return left_pose, right_pose

    # -----------------------------------------------------------------------
    # World pose access
    # -----------------------------------------------------------------------

    def get_world_pose(self) -> Tuple[Any, Any]:
        """
        Get the robot base world pose.

        Returns:
            Tuple of (position, orientation) where orientation is quaternion (wxyz).
            Backend depends on SimulationContext (numpy ndarray or torch Tensor).
        """
        self._physics_sim_view.update_articulations_kinematic()
        pose = self._root_physx_view.get_root_transforms()
        pos = pose[0, :3]
        quat = _quat_xyzw_to_wxyz(pose[0, 3:7])
        return pos, quat
