# 场景观测接口

`offense_analyzer.py` 不直接运行视觉模型，而是消费各感知模块汇总后的逐帧 JSON。
车道编号从画面左侧向右侧递增。缺失或不可靠的字段应写成 `null`、`unknown` 或低置信度，
规则引擎会放弃指控，而不是把未知状态当作违法证据。

```json
{
  "schema_version": 2,
  "source_video": "input.mp4",
  "source_fps": 10.0,
  "frame_records": [
    {
      "frame": 1,
      "timestamp_seconds": 0.0,
      "camera_pose": {
        "mode": "road_context",
        "pan_degrees": 0.0,
        "tilt_degrees": -5.0,
        "pan_rate_deg_s": 0.0,
        "tilt_rate_deg_s": 0.0,
        "calibrated": true,
        "road_transform_valid": true,
        "confidence": 0.95
      },
      "vehicles": [
        {
          "track_id": 1,
          "label": "car",
          "lane_index": 0,
          "lane_confidence": 0.95,
          "turn_signal": "off",
          "turn_signal_confidence": 0.90,
          "stop_line_position": "before",
          "signal_id": "main",
          "direction_alignment": 0.98,
          "direction_confidence": 0.90
        }
      ],
      "traffic_lights": [
        {
          "signal_id": "main",
          "state": "red",
          "confidence": 0.96,
          "applies_to_lanes": [0, 1]
        }
      ],
      "discarded_objects": [
        {
          "object_track_id": 20,
          "source_track_id": 1,
          "confidence": 0.85,
          "box": [100, 100, 120, 125]
        }
      ],
      "lane_boundary_crossings": [
        {
          "track_id": 1,
          "boundary_id": "lane_boundary_1",
          "boundary_type": "solid",
          "confidence": 0.90
        }
      ],
      "crosswalk_interactions": [
        {
          "vehicle_track_id": 1,
          "pedestrian_track_id": 11,
          "crosswalk_id": "crosswalk_1",
          "pedestrian_has_priority": true,
          "vehicle_crossed_crosswalk": true,
          "vehicle_stopped_before_crosswalk": false,
          "confidence": 0.90
        }
      ]
    }
  ]
}
```

字段约束：

- `turn_signal`：`left`、`right`、`off`、`hazard` 或 `unknown`。
- `stop_line_position`：`before`、`after` 或 `unknown`。
- `traffic_lights.state`：`red`、`yellow`、`green` 或 `unknown`。
- `signal_id` 将车辆所在车道与对应信号灯关联；没有可靠关联时不要生成闯红灯事件。
- `source_track_id` 表示抛出物轨迹已经由感知模块关联到车辆；仅检测到路面垃圾不够。
- `camera_pose.mode` 为 `road_context` 或 `target_follow`。只有前者在云台稳定、完成标定且
  `road_transform_valid` 为真时，才允许车道、红灯、实线、逆行和斑马线规则产生事件。
- `pan_rate_deg_s`、`tilt_rate_deg_s` 超过配置阈值，或云台偏角过大时，道路几何规则自动失效。
- `lane_boundary_crossings` 的实线跨越必须连续达到 `minimum_boundary_observations` 帧；虚线不触发
  `solid_line_crossing`。
- `crosswalk_interactions` 只有在行人有优先权、车辆确实越过斑马线、且没有在斑马线前停车时，才会
  产生 `pedestrian_failure_to_yield` 疑似事件。车辆、行人和斑马线必须由感知模块可靠关联。

运行：

```bash
python src/offense_analyzer.py scene_observations.json
```

默认规则参数位于 `config/offense_rules.json`。输出只表示需要人工复核的疑似事件，不是
行政处罚、事故责任或其他法律结论。
