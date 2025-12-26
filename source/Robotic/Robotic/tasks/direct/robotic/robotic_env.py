# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from tkinter.font import names
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform


from .robotic_env_cfg import RoboticEnvCfg

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

from gym import spaces
import numpy as np

class RoboticEnv(DirectRLEnv):
    cfg: RoboticEnvCfg

    def __init__(self, cfg: RoboticEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

        # action_space: 8 DOF velocity control
        # revolute joints: [-1.0, 1.0] rad/s
        # prismatic joints (gripper): [-0.2, 0.2] m/s
        low  = np.array([-1.0]*7 + [-0.2], dtype=np.float32)
        high = np.array([ 1.0]*7 + [ 0.2], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        names = self.robot.body_names
        print("Robot bodies:", names)
        self._finger_idx_pair = (names.index("grasp_3"), names.index("grasp_4"))
        print(f"[EE] using fingers midpoint: grasp_3, grasp_4")
        
        self._init_tensors_once()

        fan_world_pos = self.fan.data.root_pos_w.clone()            # [num_envs, 3]
        fan_world_quat = self.fan.data.root_quat_w.clone()          # [num_envs, 4]
        self._fan_spawn_local_pos  = fan_world_pos  - self.scene.env_origins  # local
        self._fan_spawn_local_quat = fan_world_quat.clone()                     # 世界四元數 = local（根是 env_x 原點）

        # test
        print("dt =", self.cfg.sim.dt, "decimation =", self.cfg.decimation)
        print("episode_length_s =", self.cfg.episode_length_s)
        print("max_episode_length (expected) ≈",
            int(self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation)))
        print("actual max_episode_length (env) =", int(self.max_episode_length))


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # add objects
        fan_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/fan",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.cfg.fan_usd,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, linear_damping=0.01, angular_damping=0.01,
                    max_depenetration_velocity=2.0
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.15),  # 依模型調
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=self.cfg.fan_spawn_base, rot=(1,0,0,0),
            ),
        )
        
        self.fan = RigidObject(fan_cfg)

        plate = sim_utils.UsdFileCfg(usd_path=self.cfg.plate_usd)
        plate.func("/World/envs/env_.*/plate", plate, translation=self.cfg.plate_spawn_base)

        rack = sim_utils.UsdFileCfg(usd_path=self.cfg.rack_usd)
        rack.func("/World/envs/env_.*/rack", rack, translation=self.cfg.rack_spawn_base)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=True)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["fan"]   = self.fan
        # self.scene.
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.jacobians = None

    def _init_tensors_once(self):
        self.prev_actions = torch.zeros((self.num_envs, self.action_space.shape[0]), device=self.device)
        self.prev_xy_dist = torch.zeros((self.num_envs,), device=self.device)
        self.ee_pos  = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.fan_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.fan_quat= torch.zeros((self.num_envs, 4), device=self.device)
    
    def _compute_intermediate(self):
        # EE pose
        i3, i4 = self._finger_idx_pair
        p3 = self.robot.data.body_pos_w[:, i3]
        p4 = self.robot.data.body_pos_w[:, i4]
        self.ee_pos = 0.5 * (p3 + p4) - self.scene.env_origins
        # 四元數可用其中一指或 gripper_base_2（若存在）代表
        if "TF_1" in self.robot.body_names:
            itf = self.robot.body_names.index("TF_1")
            self.ee_quat = self.robot.data.body_quat_w[:, itf]
        else:
            self.ee_quat = self.robot.data.body_quat_w[:, i3]

        # fan pose 
        self.fan_pos  = self.fan.data.root_pos_w - self.scene.env_origins
        self.fan_quat = self.fan.data.root_quat_w

        # gripper 開口（兩 slider 距離）
        idx10 = self.cfg.dof_names.index("Slider10")
        idx09 = self.cfg.dof_names.index("Slider9")
        jpos = self.robot.data.joint_pos
        self.gripper_gap = (jpos[:, idx10] - jpos[:, idx09]).abs()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        if self.actions.dim() > 2:
            self.actions = self.actions[:, 0, :]

        # --- 動作平滑，避免抖動 ---
        if not hasattr(self, "_action_smooth"):
            self._action_smooth = 0.5  # 0~1, 值越大越平滑
        self.actions = (1 - self._action_smooth) * self.actions + self._action_smooth * self.prev_actions

        joint_vels = self.robot.data.default_joint_pos.clone()
        # arm 7 joints
        arm = self.actions[:, :7]
        # gripper speed (m/s)
        g_raw = torch.clamp(self.actions[:, 7], -0.2, 0.2)

        # --- gate：距離太遠就禁止夾爪動作 ---
        # 在 _compute_intermediate 已經計了 self.ee_pos/self.fan_pos；此處先算一次距離
        rel = (self.ee_pos - self.fan_pos)
        dist = torch.linalg.norm(rel, dim=-1)      # 3D距離
        mask = (dist < self.cfg.align_thresh).float()  # 太遠就不動作
        g = g_raw * mask

        joint_vels[:, :7] = arm
        # 兩邊反向
        joint_vels[:, 7]  =  g
        joint_vels[:, 8]  = -g

        # self.robot.set_joint_effort_target(joint_vels, joint_ids=self.dof_idx)
        # self.robot.set_joint_position_target(joint_vels, joint_ids=self.dof_idx)
        self.robot.set_joint_velocity_target(joint_vels, joint_ids=self.dof_idx)

        # test: 讓 fan 緩慢移動到 EE 
        # # 1) 方向：由 fan 指向 EE
        # rel = (self.ee_pos - self.fan_pos)               # [N,3]
        # dist = torch.linalg.norm(rel, dim=-1).clamp(min=1e-8)
        # dirn = rel / dist.unsqueeze(-1)                  # 單位向量

        # # 2) 單步位移 = v * dt_env
        # # DirectRLEnv 的「控制步 dt_env」= sim.dt * decimation
        # dt_env = self.cfg.sim.dt * self.cfg.decimation
        # step = 0.02 * dt_env         # m/step

        # fan_state = self.fan.data.root_state_w.clone()      # [N,13] 或 [N,?]，前3為位置
        # fan_state[:, :3] += dirn * step
        # #（可選）保持角度不變；若想面向 EE 可在此改 fan_state[:,3:7]
        # self.fan.write_root_state_to_sim(fan_state)

    def _get_observations(self) -> dict:
        self._compute_intermediate()

        rel_pos = self.ee_pos - self.fan_pos
        # 只用相對位置 + EE 四元數 + fan 四元數 + gripper gap + prev action，簡潔即可
        obs_list = [
            rel_pos,                    # 3
            self.ee_quat,               # 4
            self.fan_quat,              # 4
            self.fan_pos,               # 3
            self.gripper_gap.unsqueeze(-1),  # 1
            self.prev_actions           # 8
        ]
        obs = torch.cat(obs_list, dim=-1)  # 維度 = 3+4+4+3+1+8 = 23
        self.cfg.observation_space = obs.shape[-1]  # 動態校正

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate()

        # === 1) 距離 ===
        rel = self.ee_pos - self.fan_pos
        dist = torch.linalg.norm(rel, dim=-1)      # 3D距離
        r_reach = -dist                            # 距離越近獎勵越高（越負越不好）

        # === 2) 夾緊成功 ===
        # close_ok = (self.gripper_gap < self.cfg.grasp_width_close_threshold)
        # near_ok  = (dist < self.cfg.align_thresh)
        # r_grasp  = (close_ok & near_ok).float()

        # # === 3) 抬起成功 ===
        # lift_ok = (self.fan_pos[:, 2] > (self.cfg.plate_spawn_base[2] + self.cfg.lift_height_thresh))
        # r_lift  = lift_ok.float() * 3.0

        # # === 動作成本 ===
        # act_pen_arm  = -0.005 * torch.linalg.norm(self.actions[:, :7], dim=-1)
        # g_speed      = self.actions[:, 7].abs()
        # far_mask     = (dist > 0.08).float()
        # act_pen_grip = -0.05 * g_speed * far_mask

        # === 總獎勵 ===
        # rew = (1.0*r_reach + 0.5*r_grasp + 1.0*r_lift + act_pen_arm + act_pen_grip)
        rew = r_reach
        # print(r_reach, r_grasp, r_lift)
        # === 狀態記錄 ===
        self.prev_actions = self.actions.clone()

        return rew



    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 可選：成功就提早終止（避免太長）
        success = (self.fan_pos[:, 2] > (self.cfg.plate_spawn_base[2] + self.cfg.lift_height_thresh))
        done = time_out | success
        return done, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # set the root state for the reset envs
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        # joints: 打開夾爪，手臂回到 default（或你自定 reset 姿）
        jpos = self.robot.data.default_joint_pos[env_ids].clone()
        # 讓夾爪張開一點（0.02 m 之類）
        jpos[:, 7] =  0.02
        jpos[:, 8] = -0.02
        jvel = torch.zeros_like(jpos)
        self.robot.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)
        self.robot.reset()

        # fan randomize
        fan_state = self.fan.data.default_root_state[env_ids].clone()
        # 位置＝(落位 local) + env_origin
        fan_state[:, 0:3] = self._fan_spawn_local_pos[env_ids] + self.scene.env_origins[env_ids]

        # 角度：沿用當時落位的方向（或保留你自己的 yaw 隨機）
        fan_state[:, 3:7] = self._fan_spawn_local_quat[env_ids]

        # 速度清零，避免解穿插把它彈走
        fan_state[:, 7:] = 0.0

        self.fan.write_root_state_to_sim(fan_state, env_ids)
        self.fan.reset()

        # 清空暫存
        self.prev_actions[env_ids] =  torch.zeros_like(self.prev_actions[env_ids])
        self._compute_intermediate()
        rel = (self.ee_pos - self.fan_pos)
        self.prev_xy_dist[env_ids] = torch.linalg.norm(rel[env_ids, :2], dim=-1)