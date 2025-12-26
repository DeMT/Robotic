import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import math
from pathlib import Path

# 這支 .py 檔案所在的位置
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[5]
ASSET_DIR = REPO_ROOT / "Robotic"

RS_M90E7A_CONFIG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/RS_M90E7A",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_DIR / "RobotLeftArm.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=8,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # rot=(1.0, 0.0, 0.0, 0.0),
        rot=(math.cos(math.radians(-25)/2), 0.0, 0.0, math.sin(math.radians(-25)/2)),
        joint_pos={
            # 7 軸
            "Revolute1": 0.0,
            "Revolute2": 0.0,
            "Revolute3": 0.0,
            "Revolute4": 0.0,
            "Revolute5": 0.0,
            "Revolute6": 0.0,
            "Revolute7": 0.0,
            # 兩個夾爪
            "Slider9": 0.0,    # 0 ~ 0.05
            "Slider10": 0.0,   # -0.05 ~ 0
        },
    ),
    
    actuators={
        "arm_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "Revolute7",
                "Revolute6",
                "Revolute5",
                "Revolute4",
                "Revolute3",
                "Revolute2",
                "Revolute1",
            ],
            effort_limit_sim=100000.0,
            velocity_limit_sim=2.0,   # rad/s
            stiffness=0.0,
            damping=40.0,
        ),
        "gripper_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "Slider9",
                "Slider10",
            ],
            effort_limit_sim=7.2,    # N*m
            velocity_limit_sim=0.3,   # m/s
            stiffness=0.0,
            damping=40.0,
        ),
    },
)

