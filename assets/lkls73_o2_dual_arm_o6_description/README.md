# LKLS73 O2 双臂 + O6 双手

新模型由以下本地源文件组合，原文件保持不变：

- `../lkls73_o2_dual_arm_description/urdf/lkls73_o2_dual_arm_description.urdf`
- `../o6/left/linkerhand_O6_left_prefixed.urdf`
- `../o6/right/linkerhand_o6_right.urdf`

## 模型入口

- `urdf/lkls73_o2_dual_arm_o6_description.urdf`：ROS 包版，网格使用 `package://lkls73_o2_dual_arm_o6_description/...`，需先把本目录作为 ROS 包构建并加载环境。
- `urdf/lkls73_o2_dual_arm_o6_description_portable.urdf`：相对路径版，网格使用 `../meshes/...`，供支持相对 URDF 路径的加载器使用；移动时保留整个目录。

## 手腕连接

| 固定关节 | 父 link | 子 link | xyz (m) | rpy (rad) |
| --- | --- | --- | --- | --- |
| Left_Wrist_to_O6_Hand_Joint | Left_Wrist_Roll_Link | lh_hand_base_link | -0.0225 0 -0.056 | 0 pi +pi/2 |
| Right_Wrist_to_O6_Hand_Joint | Right_Wrist_Roll_Link | rh_hand_base_link | -0.0225 0 -0.056 | 0 pi -pi/2 |

安装偏移根据腕部 STL 的末端平面及圆形特征推定；O6 底部安装面为局部 z=0。安装旋转采用 Rz(yaw) * Ry(pi)：左手 yaw=+pi/2，右手 yaw=-pi/2。机械臂零位时，两侧大拇指均朝机器人 +X 前方，手指仍沿 -Z 向下伸出，掌心相向。没有额外加入转接板厚度；实物装配时应以机械图纸/测量修正这两个固定关节的 origin。

## 关节与动力学参数

共 41 links、40 joints：14 个机械臂 revolute、22 个手部 revolute、4 个 fixed。10 个手部关节为 mimic，因此独立控制关节为 26 个：双臂 14 + 左手 6 + 右手 6。

原机械臂的 link、关节名、关节顺序、限位、惯量保留。左右手分别使用 `lh_`/`rh_` 前缀，mimic 保留源文件的系数与 offset。左右手网格分别放在 `meshes/left_hand` 和 `meshes/right_hand`，避免同名网格混用。

源文件左右手总质量不一致：左手 0.36998 kg，右手 0.113956201154497 kg。本模型原样保留这些参数，未凭空校准；动力学仿真前应核对实测质量及惯量。

## 解析检查命令

```bash
cd /home/tulay/桌面/urdf
source /opt/ros/jazzy/setup.bash
check_urdf lkls73_o2_dual_arm_o6_description/urdf/lkls73_o2_dual_arm_o6_description.urdf
```

`preview.png` 为新模型零位网格预览；结构及网格验证不能代替实物装配尺寸校核。
