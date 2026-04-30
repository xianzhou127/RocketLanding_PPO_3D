# RocketLanding_PPO_3D

该项目基于 **MuJoCo** 和 **PPO（Proximal Policy Optimization）** 实现火箭三维垂直回收控制，并在环境中加入了简易气动阻力模型。

## Result Preview

下面的 GIF 由 `res/rocket_PPO_3D_2.mp4` 转换得到，用于在 GitHub README 中直接展示 3D 火箭回收效果。

![RocketLanding PPO 3D demo](./res/demo.gif)

高清原始视频：

[查看 / 下载 MP4](./res/rocket_PPO_3D_2.mp4)

## Project Overview

本项目将火箭垂直回收问题建模为一个连续控制强化学习任务：

- 使用 **MuJoCo** 构建三维火箭动力学仿真环境；
- 使用 **PPO** 作为端到端强化学习控制器；
- 动作空间包含主发动机推力与二维万向节控制；
- 状态空间包含位置、速度、姿态、角速度和控制历史等信息；
- 环境中加入简易气动阻力模型，使下降过程更接近真实物理约束；
- 训练目标是让火箭在三维空间中稳定减速、姿态收敛并完成垂直回收。

## Main Files

```text
.
├── PPO.py              # PPO 算法主体
├── PPO_agent.py        # 训练 / 评估入口
├── rocket.py           # MuJoCo Gymnasium 环境封装
├── utilsPPO.py         # 网络、评估、工具函数
├── test_engine.py      # 发动机 / MuJoCo 测试
├── test_gym_env.py     # Gym 环境测试
├── rocket/             # MuJoCo XML 与 3D 模型资源
└── res/                # 结果视频与 README 展示 GIF
```

## Result Assets

```text
res/
├── demo.gif
└── rocket_PPO_3D_2.mp4
```

`demo.gif` 是为 GitHub README 展示压缩后的动图版本；`rocket_PPO_3D_2.mp4` 是原始高清录屏。

## Requirements

```bash
pip install -r requirements.txt
```

主要依赖：

- Python 3
- MuJoCo
- Gymnasium
- PyTorch
- NumPy / SciPy
- Matplotlib / TensorBoard

## Quick Start

查看 MuJoCo 模型：

```bash
python3 -m mujoco.viewer --mjcf ./rocket/rocket.xml
```

运行 PPO 训练 / 评估入口：

```bash
python PPO_agent.py
```

如果在无硬件 OpenGL 的环境中运行，可尝试：

```bash
export LIBGL_ALWAYS_SOFTWARE=1
```

## Notes

大型训练/运行产物没有上传到仓库，包括：

- `model/`
- `runs/`
- `MJMODEL.TXT`
- `*.blend1` 备份文件
- Python 缓存与 MuJoCo 运行日志

这样可以让仓库保持轻量，同时保留核心代码、MuJoCo 模型资源和结果展示文件。
