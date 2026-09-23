# 车辆危险驾驶视觉识别

本项目用于 Orange Pi 5 Pro 上的疑似恶意别车视觉识别与证据留存。

## 固定位置

- 电脑端源代码：`C:\Users\yyy12\dangerous_driving_vision`
- 香橙派运行目录：`/home/orangepi/dangerous_driving_vision`
- 香橙派连接：`ssh orangepi@10.151.177.72`
- VNC 连接：`10.151.177.72:5900`

局域网通过 DHCP 分配时，IP 地址可能变化。代码和录像目录不会因此变化。

## 当前阶段

第一阶段仅验证证据录像链路：

1. USB 摄像头以 1920x1080、30 FPS、MJPEG 输出。
2. Rockchip MPP 将画面硬件编码为 H.264，并连续写入 2 秒视频分段。
3. 程序自动清理旧分段，维持约 30 秒循环缓存。
4. 手动触发后保存触发前 10 秒和触发后 15 秒。
5. 每个事件生成原始视频、现场帧、JSON 元数据和 SHA-256 清单。

车辆检测、ByteTrack 跟踪和第一版轨迹风险分析已经加入。车道线、经过标定的
TTC、车身颜色和车牌识别仍需后续实现。

## 运行（香橙派）

```bash
cd ~/dangerous_driving_vision
python3 src/recorder.py
```

运行后输入 `t` 手动触发事件，输入 `q` 退出。

运行产生的文件位于：

```text
recordings/buffer/
recordings/events/event_<time>_<id>/
logs/ffmpeg-recorder.log
```

录像、日志、模型和数据集不会加入 Git。当前录像码率为 6 Mbps，约 30 秒事件视频通常为 20 至 25 MB。

## 单帧车辆检测

在香橙派的 RKNN 虚拟环境中运行：

    source ~/rknn_lite_env/bin/activate
    cd ~/dangerous_driving_vision
    python src/vehicle_detector.py
    python src/vehicle_detector.py --image recordings/test_inputs/road_test.png

程序仅保留 COCO 中的汽车、摩托车、公交车和卡车类别，并将测试图保存到
recordings/detection_test.jpg。录像程序和单帧检测程序当前不能同时占用摄像头；
完成单帧验证后再将两条链路合并为一个 GStreamer 管道。

当前 RKNN 模型要求 RGB、uint8、NHWC 输入，模型内部已经包含归一化。可使用
scripts/diagnose_rknn_input.py 比较输入布局、颜色顺序和数值缩放。

## ByteTrack 多目标跟踪

项目内置 Supervision 0.18.0 的 ByteTrack 核心源码，来源和许可证见
third_party/supervision_bytetrack。香橙派虚拟环境需要 SciPy 1.11.4。

对同一张道路测试图执行 10 帧跟踪：

    python src/vehicle_tracker.py --image recordings/test_inputs/road_test.png --frames 10

结果写入 recordings/tracking_test.jpg 和 recordings/tracking_test.json。

离线视频跟踪会处理完整视频，并输出带车辆框、轨迹和 ID 的 MP4：

    python src/vehicle_tracker.py --video recordings/test_inputs/input.mp4

使用 `--frames 100` 可以只处理前 100 帧，使用 `--output` 可以指定输出 MP4
路径。每帧的时间、车辆框、类别、置信度和跟踪 ID 会写入同名 JSON 文件。

## 轨迹风险分析（电脑或香橙派）

风险分析器读取跟踪 JSON，筛选“快速接近”和“快速横向移动”候选片段，不需要
RKNN、摄像头或 SciPy。新生成的跟踪 JSON 已包含画面宽高：

    python src/risk_analyzer.py recordings/tracking_test.json

对于宽高字段尚未加入之前生成的 1280x720 测试结果，使用：

    python src/risk_analyzer.py recordings/C_000002_tracked.json --frame-width 1280 --frame-height 720

输出文件默认与输入文件位于同一目录，文件名以 `_risk.json` 结尾。候选事件只用于
定位需要人工复核的时间段，不能单独作为危险驾驶或法律责任结论。实际 TTC 仍需完成
摄像头内参和安装姿态标定。

## 导出人工复核片段（电脑或香橙派）

根据风险报告从带跟踪框的视频中截取短片，默认保留事件前后各 1 秒；重叠片段会自动
合并：

    python src/review_clip_exporter.py tracked.mp4 tracked_risk.json --output-dir recordings/review_clips

输出目录包含 `review_*.mp4` 和 `review_manifest.json`。这些文件属于生成结果，不会
加入 Git。

## 从 Windows 部署到香橙派

先提交电脑端代码，再在 Windows PowerShell 中运行以下单行命令。将 IP 替换为香橙派
当时在局域网内的实际地址：

    powershell -ExecutionPolicy Bypass -File .\scripts\deploy_orangepi.ps1 -OrangePiHost 10.151.177.72

脚本仅上传当前 Git 提交中的源代码和文档，并保留香橙派已有的录像、模型、数据集和
虚拟环境。工作区存在未提交改动时，脚本会拒绝部署。

## 多类疑似违法行为分析

`src/offense_analyzer.py` 已实现以下时序规则：

- 车辆完成变道，但变道前后没有检测到对应方向的转向灯；
- 同一车辆在设定时间内连续完成两次或更多变道；
- 对应车道信号灯稳定为红灯时，车辆从停止线前方移动到后方；
- 与车辆关联的抛出物轨迹连续出现多帧。

运行命令：

    python src/offense_analyzer.py scene_observations.json

输入字段和置信度要求见 `docs/scene_observation_schema.md`，规则阈值位于
`config/offense_rules.json`。当前 YOLOv8n 车辆模型只能提供车辆框；车道线、转向灯、
信号灯与停止线关联、抛出物检测仍需要对应的视觉模型，不能由规则引擎凭空推断。
