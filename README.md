# Project09：双臂 RGB-D 视觉接球与 O6 灵巧手策略

当前包保留第1446次更新的 `model_1445.pt`、完整优化器状态、Isaac训练源码及MuJoCo播放器。
任务是接住来自正面/侧面的40mm球，物理接触后闭合手指，连续保持1秒并包含倾斜保持。
**双手连续互抛尚未实现；本包不是已收敛策略或实机部署版本。**

## 快速播放 MuJoCo

验证环境为Linux、Python3.12、MuJoCo3.11.0、PyTorch2.11.0，CPU推理，显示需要OpenGL。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mujoco.txt
python scripts/play_mujoco_policy.py \
  --checkpoint checkpoints/model_1445.pt \
  --camera_resolution 192 --seed 42 --steps 36000
```

关闭窗口即可退出。模型和缓存首次运行时自动生成；可用鼠标旋转、平移和缩放视角。
也可使用 `bash scripts/run.sh play-mujoco --checkpoint checkpoints/model_1445.pt`。
无窗口检查：

```bash
python scripts/play_mujoco_policy.py --headless \
  --checkpoint checkpoints/model_1445.pt --steps 3600
python -m unittest discover -s tests
```

## 保留的内容

- `checkpoints/model_1445.pt`：Actor、Critic、观测归一化、Adam状态和已完成更新数1446。
- `checkpoints/actor.pt`：冻结TorchScript均值策略，发布验证时导出，输入100维、输出14维。
- `checkpoints/source/config/`、runner_config、kinematic_contract：同源参数与关节顺序。
- `assets/`：双臂及O6手URDF/STL、训练同源mirror_left惯性版本。
- `scripts/`、`tests/`：感知、预测拦截、接触抓握、PPO训练及MuJoCo适配代码与回归测试。
- `reports/`：最新迁移证据、包验证；历史中间权重和训练日志不在包内。
- `PACKAGE_MANIFEST.json`、`SHA256SUMS`：逐文件大小、校验值及原始权重哈希。

## 策略与控制链

渲染RGB-D → 红球像素与球面拟合 → 带重力的弹道滤波 → 预测落点/到达时间 →
雅可比手臂跟踪 + PPO残差 → 真实接触触发闭合 → 连续与倾斜保持。
球真值仅用于回合重置、结果判定和误差诊断，不代替相机输入策略或拦截规划。

观测为26关节位置差、26关节速度、视觉球相对掌心位置3、球速度3、可见性1、观测年龄1、
接球手one-hot2、阶段2、手指接触10、历史基础动作8，加18维拦截/速度/历史修正特征，共100维。
每只手动作顺序为 `[rx, ry, z, closure, x, y, preclose_time]`，双手共14维。
XY每动作单位20mm、每轴上限±200mm，控制器保留关节与驱动约束。

训练使用256并行、192×192 RGB-D、120Hz物理、60Hz控制；0/1/2级按回合等概率混合。
训练方向目标为正面35%、左侧25%、右侧40%。学习率固定0.00005，PPO每批3个epoch。
MuJoCo播放使用完整前/侧来球分布和确定性均值动作，不加训练用感知扰动。

## 验证边界

发布前源码有107项测试通过，包含归一化、策略导出、控制指令、质量、掌心FK和雅可比检查。
原RSL-RL与迁移Actor在512组输入上的最大输出误差0，未修改原始权重。
MuJoCo3600步运行检查：34回合成功4回合，约11.76%，视觉球心平均误差约3.1mm，物理异常0。
原生窗口控制循环约60帧/秒、1倍实时；这些是运行检查，不是独立成功率验证。
**当前第1446次权重的完整Isaac验证在用户停止时尚未完成，不提供虚构的完整成功率。**

MuJoCo与PhysX的碰撞、mimic、摩擦和驱动语义存在差异，跨引擎成功率尚未对齐。
未绑定球、添加橡胶板、放宽保持标准或重写飞行速度来提高显示成功率。
详见[迁移说明](docs/mujoco_policy_playback.md)与[来源边界](NOTICE.md)。

## Isaac 训练与播放

保留原Isaac Lab3.0EA / Isaac Sim6.1源码；仅安装MuJoCo依赖不包含Isaac运行环境。
使用自己的Isaac Lab Python设置 `ISAAC_PYTHON`。Isaac启动还要求用户接受NVIDIA Omniverse EULA。

```bash
export ISAAC_PYTHON=/path/to/isaaclab/python
# 接受NVIDIA协议后才设置：export OMNI_KIT_ACCEPT_EULA=YES
bash scripts/run.sh play-policy --play_rebind_views --policy_version v4 \
  --response_profile fast --checkpoint checkpoints/model_1445.pt \
  --device cuda:0 --num_envs 1 --camera_resolution 192 --eval_steps 36000
```

训练续训使用 `scripts/train_policy_v2.py --initial_checkpoint checkpoints/model_1445.pt`，
并显式指定 `--training_profile mixed_all --tracking_profile predictive --grip_profile pose_finish` 和新输出目录。
正常播放命令不训练，也不自动恢复后台训练。
