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

摄像头安装在前挡风玻璃附近并朝向车外时，二维云台使用两种模式：`road_context` 用于稳定前视，
建立车道、停止线、红绿灯、斑马线和道路边界的几何关系；`target_follow` 用于短时跟随目标，补拍
车辆、转向灯和车牌等细节。云台快速转动或道路坐标变换无效时，系统会禁止依赖固定道路几何的
规则，避免把云台运动误判为变道、压线或闯红灯。每帧应记录云台角度、角速度、模式和标定状态。

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

### 锁定一辆目标车

跟踪 ID 是第一遍跟踪产生的编号。需要锁定目标时，先完成普通跟踪，查看同名
JSON 或标注视频中的 `track_id`，再用该 ID 重跑：

    python src/vehicle_tracker.py --video recordings/test_inputs/input.mp4 \
        --target-track-id 12 --lock-confirm-frames 3 --target-max-missed 10 \
        --output recordings/input_target12.mp4

也可以按第一帧画面坐标选择车辆（坐标原点在左上角）：

    python src/vehicle_tracker.py --video recordings/test_inputs/input.mp4 \
        --target-point 960,540 --output recordings/input_target.mp4

目标必须连续出现 `--lock-confirm-frames` 帧才会变成 `locked`。短时检测失败时，
状态变为 `temporarily_lost`，程序最多保持 `--target-max-missed` 帧；重新关联必须满足
位置和 IoU 条件，两个候选同样接近时保持原目标并等待，不会随意换车。超过容忍帧数
后状态为 `lost`，不会自动锁定另一辆车。输出视频会用红框标出已锁定目标，JSON 每帧
包含 `target_lock`（状态、目标 ID、框、丢失帧数和匹配方式）。

需要更保守地防止换车时，可显式设置：

    --reassociation-confirm-frames 2 --max-reassociation-gap 3 \
    --target-min-match-score 0.35 --target-min-match-margin 0.10 \
    --target-max-area-change 2.25

其中 `max-reassociation-gap` 限制允许尝试换 ID 的丢失帧数；超过该值直接进入
`lost`。可靠性优先时应保持较小的值，宁可报告目标丢失，也不要自动跳到另一辆车。

要锁定“疑似违法车辆”，先运行普通跟踪和 `risk_analyzer.py`，从风险报告的事件中
人工确认 `track_id`，再用 `--target-track-id` 重跑。运动风险候选不能直接证明违法；
压实线、闯红灯、未礼让、不打灯等规则还必须有经过标定的车道、信号灯、转向灯和
行人观测，最终结果仍需人工复核。

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

中国道路素材和公开数据源登记在 `config/china_video_catalog.json`。来源审计：

已核验的开源组件、固定提交版本、许可证和能力缺口登记在
`config/open_source_components.json`，详细筛选结论见
`docs/open_source_traffic_violation_references.md`。PaddleDetection
PP-Vehicle 的压线、逆行和车道数据处理源码已带许可证保存在
`third_party/traffic_violation_references/paddledetection_ppvehicle_apache2`。

    python scripts/audit_video_catalog.py --no-hash

重新检索 GitHub 候选来源（只登记搜索结果，不自动下载）：

    python scripts/search_public_sources.py --output recordings/public_source_search.json

批量运行已有本地中国道路评估片段：

    python scripts/run_china_validation_suite.py --output-dir D:/桌面/dangerous_driving_test/china_validation_results --dry-run

只跑新增的开放许可外部验证集：

    python scripts/run_china_validation_suite.py --output-dir recordings/online_validation --id commons_chinese_signal_cc0 --id commons_china_zebra_collision_pd --device 0 --scene-context

比较候选交通跟踪参数时显式增加：

    --tracker config/bytetrack_traffic.yaml

公开数据集用于补充车辆、车道、交通灯和异常场景感知；它们没有自动产生本项目七类违法标签。公开视频或新闻页面默认只能做人工评估，只有明确许可或自采并授权的数据才可训练。

`src/offense_analyzer.py` 已实现以下时序规则：

- 车辆完成变道，但变道前后没有检测到对应方向的转向灯；
- 同一车辆在设定时间内连续完成两次或更多变道；
- 对应车道信号灯稳定为红灯时，车辆从停止线前方移动到后方；
- 车辆连续跨越实线或双实线；
- 车辆在稳定道路上下文中持续逆向行驶；
- 行人拥有斑马线优先权时，车辆越过斑马线且未在前方停车；
- 与车辆关联的抛出物轨迹连续出现多帧。

运行命令：

    python src/offense_analyzer.py scene_observations.json

输入字段和置信度要求见 `docs/scene_observation_schema.md`，规则阈值位于
`config/offense_rules.json`。当前 YOLOv8n 车辆模型只能提供车辆框；车道线、转向灯、
信号灯与停止线关联、道路边界、行人优先权和抛出物检测仍需要对应的视觉模型，不能由规则引擎
凭空推断。所有输出都是疑似违法行为和证据片段，必须人工复核，不能直接作为法律定责。

### 当前可运行的道路几何离线验证

`src/road_offense_pipeline.py` 可将车辆跟踪 JSON 与人工标定的道路几何组合，生成规则引擎
所需的场景观测，并验证实线、斑马线和信号灯规则。先用 `vehicle_tracker.py` 对视频
生成跟踪 JSON，再复制 `config/road_geometry.example.json`，按该视频画面修改标定文件：

    python src/vehicle_tracker.py --video recordings/test_inputs/input.mp4 \
        --output recordings/input_tracked.mp4
    python src/road_offense_pipeline.py recordings/input_tracked.json \
        --geometry config/road_geometry.json

实际跟踪 JSON 文件名以程序输出为准。`normalized: true` 时坐标使用 0 到 1 的画面比例。
前视道路可按画面从左到右标定 `boundaries`；固定路口或斜向道路应使用 `lanes` 多边形。
车道可配置图像坐标中的允许行驶方向 `direction: [dx, dy]`，反向或横向轨迹不会绑定该车道的
信号灯。多边形重叠处会返回未知车道，避免歧义区域产生结论。`traffic_signals` 还需标定灯框
ROI、适用车道及有方向的停止线。输出的 `*_scene_observations.json` 和 `*_offenses.json` 默认写在跟踪 JSON 所在目录；
后者包含 `selected_offender`，无满足条件的规则事件时状态为 `NO_CONFIDENT_OFFENDER`。

上海路口负样本标定位于 `config/validation/chinese_signal_cc0_geometry.json`：画面中的绿灯通行车辆
能够关联信号灯，横向车流不关联，运行结果应为零个闯红灯事件。该文件仅适用于对应素材和画面裁剪。

此验证器不自动检测车道线，也不适用于摄像头平移、俯仰或云台跟随期间直接套用固定画面标定。
车道线被遮挡、透视变化明显或边界标错时，不应采信候选结果。转向灯、红绿灯、停止线、行人
与斑马线、逆行方向及抛出物仍需可靠的感知观测；没有这些观测时对应规则不会产生结论。输出仅
用于视频复核与研发验证，不构成违法认定或执法证据。
