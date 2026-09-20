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

车辆检测、跟踪、车道线、TTC、车身颜色和车牌识别将在录像链路验证后加入。

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
