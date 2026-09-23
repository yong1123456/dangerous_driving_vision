# 场景观测接口

`offense_analyzer.py` 不直接运行视觉模型，而是消费各感知模块汇总后的逐帧 JSON。
车道编号从画面左侧向右侧递增。缺失或不可靠的字段应写成 `null`、`unknown` 或低置信度，
规则引擎会放弃指控，而不是把未知状态当作违法证据。

```json
{
  "schema_version": 1,
  "source_video": "input.mp4",
  "source_fps": 10.0,
  "frame_records": [
    {
      "frame": 1,
      "timestamp_seconds": 0.0,
      "vehicles": [
        {
          "track_id": 1,
          "label": "car",
          "lane_index": 0,
          "lane_confidence": 0.95,
          "turn_signal": "off",
          "turn_signal_confidence": 0.90,
          "stop_line_position": "before",
          "signal_id": "main"
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

运行：

```bash
python src/offense_analyzer.py scene_observations.json
```

默认规则参数位于 `config/offense_rules.json`。输出只表示需要人工复核的疑似事件，不是
行政处罚、事故责任或其他法律结论。
