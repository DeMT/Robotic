# IsaacLab environment for pick-and-place task

## Installation

1. Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

2. Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

3. Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    cd Robotic
    python -m pip install -e source/Robotic
  
4. Download the USD files from Google Drive, unzip them, and place them in the project root directory.
  
    ```bash
    Robotic/
    ├── scripts/
    │   ├── rl_games/
    │   │   ├── train.py
    │   │   └── play.py
    │   ├── list_envs.py
    │   ├── view_env.py                            (Random Agent 測試環境)
    ├── source/
    |   └── Robotic/
    │       └── Robotic/
    │           ├── robots/
    │           │   └── RS_M90E7A.py               (左手臂的配置檔，目前是用速度控制)
    │           ├── tasks/
    │           │   └── direct/
    │           │       └── robotic/
    │           │           ├── robotic_env.py     (如何跟環境互動，獎勵訊號)
    │           │           └── robotic_env_cfg.py (初始要用到的一些參數值)
    ├── Fan.usd
    ├── Plate.usd
    ├── Rack.usd
    └── RobotLeftArm.usd
    ```

5. Verify that the extension is correctly installed by:

    - Listing the available tasks:

      Note: It the task name changes, it may be necessary to update the search pattern "Template-"
      (in the `scripts/list_envs.py` file) so that it can be listed.

      ```bash
      # Under the outermost Robotic/ directory
      # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
      python scripts/list_envs.py
      ```

    - Launch a random agent:

      ```bash
      # Under the outermost Robotic/ directory
      # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
      python scripts/view_env.py
      ```

6. Train an RL agent

    ```bash
    # Under the outermost Robotic/ directory
    python scripts/rl_games/train.py --task=Template-Robotic-Direct-v0 \
      agent.params.config.horizon_length=128 \
      agent.params.config.minibatch_size=256 \
      --video \
      --video_length=1000 \
      --num_envs=20 \
      --max_iterations=500

7. Play with checkpoint

    ```bash
    python scripts/rl_games/play.py --task=Template-Robotic-Direct-v0 --checkpoint=
    ```

## Docs

- Environment Overview

  | Component           | Description                                                         |
  | ------------------- | ------------------------------------------------------------------- |
  | **Robot**           | 7-DOF manipulator (RS_M90E7A) with dual sliders for gripper control |
  | **Objects**         | Rigid bodies loaded via USD files (`Fan`, `Plate`, `Rack`)          |
  | **Scene**           | Includes ground plane, dome light, and physics-based interactions   |
  | **Action Type**     | Continuous velocity control (`Box[-1,1]^7 + gripper [-0.2, 0.2]`)   |
  | **Simulation Step** | `dt = 1/120 s`, with decimation factor of 2                         |
  | **Number of Envs**  | Configurable (`--num_envs` argument)                                |

- Observation Space

  | Feature                         | Dimension | Description                                    |
  | ------------------------------- | --------- | ---------------------------------------------- |
  | Joint positions                 | 7         | Robot arm revolute joints                      |
  | Joint velocities                | 7         | Angular velocity of each joint                 |
  | Gripper position                | 1         | Linear position of the slider joint            |
  | **Fan position**                | 3         | `(x, y, z)` world position of the `Fan` object |
  | (optional) Plate/Rack positions | 3 each    | Positions of other rigid objects               |

- Action Space

  | Type       | Dimension | Range                                          | Description                                     |
  | ---------- | --------- | ---------------------------------------------- | ----------------------------------------------- |
  | Continuous | 8         | `[-1,1]^7` (revolute) + `[-0.2,0.2]` (gripper) | Joint velocity targets applied to the robot arm |

- Reward

  | Name       | formula                           | scale | Description                                                 |
  | ---------- | --------------------------------- | ----- | ----------------------------------------------------------- |
  | r_reach    |  \|\|ee_pos - fan_pos\|\|         | -1    | Distance error between the end-effector gripper and the fan |
  | r_grasp    | 𝟙[grasp_confirmed]                | 200   | Sparse reward when the fan is successfully grasped and held |


- Success Matrix

  | Name       | formula                           | Description                                                 |
  | ---------- | --------------------------------- | ----------------------------------------------------------- |
  | s_reach    |  \|\|ee_pos - fan_pos\|\| < 0.15  | Whether the end-effector gripper is close enough to the fan |
  | s_grasp    | grasp_confirmed                   | Whether the fan is successfully grasped and held    |
