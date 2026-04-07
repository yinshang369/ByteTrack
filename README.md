# ByteTrack 工程化复现与二次开发

## 1. 项目简介
本项目基于开源 **ByteTrack** 进行复现与工程化改造，目标是把论文/开源代码中的跟踪能力，整理成更易运行、可观测、可扩展的工程脚本。

说明：
- 这是在开源项目基础上的二次开发，不是从零实现跟踪算法。
- 重点工作在于推理链路打通、输出规范化、性能统计、ONNX/ONNX Runtime 部署支持。

## 2. 功能特性
- 支持 PyTorch 视频跟踪推理，输出带 `track id` 可视化视频。
- 支持通过命令行传入任意 `mp4` 路径。
- 推理后自动生成统计信息：
  - 平均 FPS
  - 平均每帧耗时（latency）
  - 总帧数
  - 每帧检测目标数
  - 每帧有效轨迹数
- 支持 ONNX 导出（可指定权重、输入尺寸、导出路径）。
- 支持 ONNX Runtime 视频推理独立脚本（弱耦合、便于部署验证）。

## 3. 项目结构
```text
ByteTrack/
├── tools/
│   ├── demo_track.py            # PyTorch 推理主入口（已增强输出目录与统计）
│   ├── export_onnx.py           # ONNX 导出脚本（已增强参数与 IO 信息打印）
│   └── infer_onnx_video.py      # ONNX Runtime 视频推理脚本
├── yolox/
│   ├── tracker/byte_tracker.py  # 跟踪核心逻辑
│   ├── utils/visualize.py       # 可视化绘制
│   └── ...
├── exps/example/mot/            # 实验配置
├── videos/                      # 示例视频
└── outputs/                     # 推理输出（运行后生成）
```

## 4. 环境准备
建议使用 Python 3.8+，并在 Linux 环境下运行。

```bash
git clone https://github.com/yinshang369/ByteTrack.git
cd ByteTrack
pip install -r requirements.txt
pip install onnx onnxruntime onnxsim
```

如使用 GPU 推理，请按你的 CUDA 版本安装对应的 PyTorch。

## 5. 快速开始
先准备：
- 视频文件：`/path/to/demo.mp4`
- 模型权重：`pretrained/bytetrack_x_mot17.pth.tar`
- 实验配置：`exps/example/mot/yolox_x_mix_det.py`

直接运行 PyTorch 推理（会输出可视化视频和统计文件）：

```bash
python3 tools/demo_track.py video \
  --path /path/to/demo.mp4 \
  -f exps/example/mot/yolox_x_mix_det.py \
  -c pretrained/bytetrack_x_mot17.pth.tar \
  --fp16 --fuse --save_result \
  --output_dir outputs/demo_runs
```

## 6. PyTorch 推理示例
```bash
python3 tools/demo_track.py video \
  --path videos/palace.mp4 \
  -f exps/example/mot/yolox_x_mix_det.py \
  -c pretrained/bytetrack_x_mot17.pth.tar \
  --fp16 --fuse --save_result \
  --output_dir outputs/demo_runs \
  --track_thresh 0.5 \
  --match_thresh 0.8
```

## 7. ONNX 导出示例
说明：当前稳定导出的是**检测模型前向**，跟踪器（跨帧状态机）不在 ONNX 图内。

```bash
python3 tools/export_onnx.py \
  -f exps/example/mot/yolox_s_mix_det.py \
  --weights pretrained/bytetrack_s_mot17.pth.tar \
  --input-size 608,1088 \
  --output-path outputs/onnx/bytetrack_s_608x1088.onnx \
  --no-onnxsim
```

导出后会打印输入/输出 tensor 的基础信息（name / dtype / shape）。

## 8. ONNX Runtime 推理示例
```bash
python3 tools/infer_onnx_video.py \
  --video_path /path/to/demo.mp4 \
  --onnx_path outputs/onnx/bytetrack_s_608x1088.onnx \
  --output_dir outputs/onnx_runs \
  --score_thr 0.1
```

可选参数：
- `--nms_thr`
- `--input_size`
- `--providers`（如 `CPUExecutionProvider` 或 `CUDAExecutionProvider,CPUExecutionProvider`）

## 9. 输出结果说明
### PyTorch 推理输出
```text
outputs/demo_runs/{timestamp}/
├── xxx.mp4                      # 带 track id 的可视化视频
└── metrics/
    ├── metrics.json             # 逐帧统计 + 汇总
    └── summary.txt              # 简洁摘要
```

### ONNX Runtime 推理输出
```text
outputs/onnx_runs/{timestamp}/
├── xxx.mp4
└── metrics/
    ├── metrics.json
    └── summary.txt
```

## 10. 性能统计说明
统计口径（当前实现）：
- `latency_ms`: 单帧端到端处理耗时（含前处理、推理、后处理、跟踪与可视化）。
- `fps`: 单帧即时 FPS（`1 / frame_latency`）。
- `avg_fps`: 全视频平均 FPS（`total_frames / total_time`）。
- `avg_latency_ms`: 全视频平均每帧耗时。
- `detections`: 每帧检测框数量。
- `valid_tracks`: 每帧通过过滤后的有效轨迹数量。

建议对比方式：
1. 同一视频、同一阈值分别跑 PyTorch 与 ONNX Runtime。
2. 对比 `summary.txt` 中的 `avg_fps`、`avg_latency_ms`、`total_frames`。
3. 抽查同帧可视化结果，观察轨迹数量与 ID 稳定性差异。

## 11. 后续可优化方向
- 支持批量视频推理与统一报告汇总。
- 细分统计口径（preprocess / model / postprocess / tracking 各阶段耗时）。
- 增加自动化回归脚本（精度与速度双维度）。
- 增加更完善的参数配置管理（如 YAML 配置）。
- 为 ONNX Runtime 增加更稳健的 provider 自动回退与告警提示。

## 二次开发贡献点
- 基于开源 ByteTrack 完成工程化复现，不改算法核心前提下提升可运行性与可观测性。
- 新增统一输出目录与结构化统计，便于实验对比和性能分析。
- 增强 ONNX 导出工具链，支持更灵活的 CLI 参数与导出后模型信息检查。
- 新增独立 ONNX Runtime 视频推理脚本，形成 PyTorch/ONNX 两条可对照推理链路。
