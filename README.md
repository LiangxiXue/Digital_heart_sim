# Digital Heart — 二维 FitzHugh-Nagumo 心肌电传播仿真

在二维网格上数值求解 **FitzHugh-Nagumo** 反应扩散方程，模拟膜电位在组织中的传播，并导出「正常心跳」与「心律失常」对照动画。

**仓库地址：** [github.com/LiangxiXue/Digital_heart_sim](https://github.com/LiangxiXue/Digital_heart_sim)

## 功能概览

- **核心模型**：膜电位 \(v\) 与恢复变量 \(w\) 的耦合 PDE；扩散项作用于 \(v\)（扩散系数 \(D\)）。
- **数值方法**：时间方向为前向 **Euler**；空间 Laplacian 为 **五点差分**；边界为 **Neumann（零通量）**（`np.pad` 边缘延拓）。
- **三种场景**（在 `digital_heart.py` 的 `main()` 中依次运行）：
  1. **正常传播**：左上角局部起搏，扇形波前扩展 → `normal_heartbeat.gif`
  2. **螺旋波再入**：S1–S2 交叉场刺激 → `arrhythmia_heartbeat.gif`
  3. **传导阻滞**：降低扩散 + 中部坏死带 → `arrhythmia_block.gif`

## 环境要求

- **Python 3.10+**（推荐 3.11）
- 依赖见 [`requirements.txt`](requirements.txt)：`numpy`、`matplotlib`、`pillow`（导出 GIF 需要 Pillow）

## 安装与运行

```bash
git clone https://github.com/LiangxiXue/Digital_heart_sim.git
cd Digital_heart_sim

python -m pip install -r requirements.txt
python digital_heart.py
```

运行结束后，当前目录会生成上述三个 GIF。若需 **MP4**，可在 `make_animation()` 中把输出扩展名改为 `.mp4`，并确保系统已安装 **ffmpeg**（matplotlib 动画 writer）。

## 默认参数（作业模板）

| 符号 | 含义 | 默认值 |
|------|------|--------|
| \(N\) | 网格边长 | 100 |
| \(D\) | 扩散系数 | 1.0 |
| \(\varepsilon\) | 恢复时间尺度 | 0.08 |
| \(\beta\), \(\gamma\) | 恢复方程参数 | 0.7, 0.8 |
| \(dt\), \(dx\) | 时间步、空间步 | 0.1, 1.0 |

稳定性需满足近似 CFL：\(D \cdot dt / dx^2 < 0.25\)（二维）。

## 项目结构

```
Digital_heart_sim/
├── digital_heart.py    # 仿真与动画主脚本
├── requirements.txt    # Python 依赖锁定范围
├── REPORT_DRAFT.md     # 实验报告草稿（模型与参数说明）
├── README.md           # 本说明
└── *.gif               # 运行生成的示例动画（可选提交）
```

## 自定义与扩展

- 修改 `FHNParams` 或各 `scenario_*()` 中的参数即可做对照实验。
- `run_simulation(..., extra_stim_fn=...)` 支持按时间步注入额外刺激（如 S2）。

## 许可证

如无特别说明，代码按学习/作业用途使用；若需开源许可证可自行补充 `LICENSE` 文件。
