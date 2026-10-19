# 项目09当前策略的MuJoCo迁移播放

使用 `model_1445.pt`（完成1446次更新），只推理播放，不训练、不改原始权重。
Isaac训练继续保持 `paused_by_user`，原Isaac播放窗口已关闭。

## 播放命令

```bash
cd .
bash scripts/run.sh play-mujoco \
  --checkpoint "./checkpoints/model_1445.pt" \
  --camera_resolution 192 --seed 42 --steps 36000
```

当前使用已安装MuJoCo 3.11.0的 `isaaclab61` Python环境，无需启动Isaac或Kit。
默认打开原生MuJoCo交互窗口，隐藏左右编辑面板，按60Hz墙钟速度播放。
鼠标可以旋转、平移和缩放视角；关闭窗口结束播放。
可加 `--headless --steps 3600` 运行无窗口检查，日志目录默认自动创建。
运行状态见 `reports/active_playback.json`；其run_dir中包含migration.json、status.json、episodes.jsonl、actor.pt和相机截图。

## 实现与接口

- `scripts/build_mujoco_catch.py` 从训练同源 `robot_mirror_left.urdf` 生成固定基座模型，保留完整质量/惯性精度。
  36个机器人关节、26个独立位置驱动、10个手部mimic关节约束，加一个40mm自由球。
- `scripts/mujoco_catch.py` 按训练时的关节与刚体名称映射MuJoCo索引，四元数wxyz转换为策略使用的xyzw。
  直接复用ball_catch_env.py中的 `_command` 与 `get_observations` 两个纯函数；AST只加载这两个函数，不导入Isaac。
- Actor为原128/128/64 ELU网络，100维观测→14维动作；保留均值、标准差和epsilon=0.01。
  与安装版本RSL-RL MLPModel比较后导出冻结TorchScript，正常播放只执行确定性均值动作。
- 机身上方192×192 RGB-D实际渲染，使用原球面拟合与滤波器；真值仅用于抛球重置、结果判断及诊断误差。
- 保留预测拦截、预收拢、真实手球接触触发闭合和倾斜保持，未对球做焊接或位置跟随。
- 物理120Hz、感知/策略60Hz；每轴±200mm策略修正、原动作平滑和控制器限位保持。
- 播放使用现有完整前/侧来球分布，不应用训练用感知噪声。

## 已验证与限制

107项单元/回归测试通过，包括策略归一化/输出、导出模型、控制指令、质量、掌心FK与雅可比及驱动速度饱和。
左右待机掌心FK误差约1e-16m；策略与RSL-RL输出误差记录在migration.json。
真实RGB-D短测试已触发手球接触、闭合和完整保持成功；长运行检查见对应smoke目录。
这些检查证明迁移链路可运行，不代表跨引擎成功率已经等价。

MuJoCo与PhysX的接触求解、隐式驱动和速度约束实现不同：
原碰撞STL在MuJoCo中采用凸包，PhysX采用凸分解；mimic在MuJoCo中为有限刚度等式约束。
原手部网格的MuJoCo摩擦参数为滑动1.0、扭转0.005、滚动0.0001；不能将PhysX静/动摩擦直接逐项复制。
MuJoCo适配层按臂3rad/s、指1rad/s对PD期望速度饱和，保持原控制目标与力矩限位，不直接改写关节状态。
外力/接触冲量仍可能使瞬时速度超出该目标范围；这不是完全相同的PhysX绝对速度约束。
未为提高视觉成功率增加板子、放宽保持标准、绑定球或重写飞行速度。
本轮仍为外部来球接住并保持，尚未加入双手连续互抛。

接口与接触语义参考：
[MuJoCo XML reference](https://mujoco.readthedocs.io/en/3.1.3/XMLreference.html)、
[MuJoCo Python接口](https://mujoco.readthedocs.io/en/stable/python.html)。

## 本次实际运行结果

2026-10-19原生窗口 `MuJoCo : Project09_current_policy` 已启动；控制循环约60帧/秒、约1倍实时速度，物理异常0。
原RSL-RL与冻结Actor在512组100维输入上输出最大误差0；原始checkpoint SHA256保持不变。
3600步无窗口检查为34回合成功4回合（约11.76%），视觉平均误差约3.1mm，0物理异常。
这是迁移运行检查，不能与Isaac固定1024回合成功率直接比较，也未证明跨引擎效果等价。
MuJoCo模型保存在各次播放目录ball_catch.xml中，相关参数和局限记录在plant_contract.json。
完整报告：reports/mujoco_migration_1446_20261019.json。
