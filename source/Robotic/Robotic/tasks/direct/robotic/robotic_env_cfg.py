from Robotic.robots.RS_M90E7A import RS_M90E7A_CONFIG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from pathlib import Path

# 這支 .py 檔案所在的位置
HERE = Path(__file__).resolve()

# 依你的目錄結構往上 / 組出 Robotic 資料夾
# 比方說這支檔案在 <repo_root>/source/configs/my_cfg.py
# 那 repo_root 就是 HERE.parents[2]
REPO_ROOT = HERE.parents[7]
ASSET_DIR = REPO_ROOT / "Robotic"

@configclass
class RoboticEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    # - spaces definition
    action_space = 4
    observation_space = 19
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = RS_M90E7A_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=20, env_spacing=4.0, replicate_physics=True)
    dof_names = [
        'Revolute7','Revolute6','Revolute5','Revolute4','Revolute3','Revolute2','Revolute1', # Arm
        'Slider9','Slider10'
    ]    
    # gripper 幾何（粗估，訓練用）
    gripper_half_thickness: float = 0.01    # 手指厚度的一半（碰撞近似用）
    grasp_width_close_threshold: float = 0.005  # 視為「夾緊」的寬度（m）
    
    # object & randomization
    fan_usd: str   = str(ASSET_DIR / "Fan.usd")
    plate_usd: str = str(ASSET_DIR / "Plate.usd")
    rack_usd: str  = str(ASSET_DIR / "Rack.usd")
    robot_description_path: str = str(ASSET_DIR / "robot_description.yaml")
    robot_urdf_path: str = str(ASSET_DIR / "RS-M90E7A.urdf")
    
    fan_spawn_base: tuple[float,float,float] = (-0.3, -0.8, 0.0)
    plate_spawn_base: tuple[float,float,float] = (-0.3, -0.8, 0.0)
    rack_spawn_base: tuple[float,float,float]  = (-0.3, -0.8, 0.0)

    # 隨機擾動
    fan_pos_noise_xyz: tuple[float,float,float] = (0.00, 0.00, 0.0) # XY 亂數擺放
    fan_yaw_deg_range: float = 180.0

    # 成功判準
    lift_height_thresh: float = 3   # 抬離基座高度
    align_thresh: float = 0.25      # EE 與 fan  偏差門