# 中国道路视频素材清单

更新时间：2026-10-13

## 使用原则

来源分为三类：`evaluation_only` 只用于人工复核；`training_candidate` 只有在数据集条款明确允许训练后才可进入训练；`manual_review_candidate` 只有网页线索，不代表已下载或取得再利用许可。运行
`python scripts/audit_video_catalog.py --no-hash` 可检查本地文件、FFprobe 元数据和训练许可状态。
`python scripts/search_public_sources.py --output recordings/public_source_search.json` 可重新查询 GitHub 候选仓库；检索结果不会自动下载或加入训练集。
审计还会按 `source_group`、SHA-256 和来源 URL 检查训练、验证、保留测试之间的数据泄漏；`--strict` 模式遇到跨集合复用会返回失败。

## 已保存视频

以下视频位于 `D:\桌面\新建文件夹`，均能确认是中国道路场景。它们是带浏览器界面、标题和字幕的屏幕录制，适合功能验证，不应在未核实原始来源许可前作为训练数据。

| 文件 | 可验证目标 | 当前结果 | 主要限制 |
|---|---|---|---|
| `屏幕录制 2026-09-29 195716.mp4` | 压实线变道 | 多辆车可持续跟踪 | 大标题遮挡道路上部，需人工标定实线和违法车辆 |
| `屏幕录制 2026-09-29 195839.mp4` | 逆向行驶 | 当前模型基本检测失败 | 夜间强光、车体很小，需要夜间模型或图像增强 |
| `屏幕录制 2026-09-29 195920.mp4` | 不礼让行人 | 检出 5 条行人轨迹、42 次行人观测 | 需标定斑马线，并建立车辆与行人的时序关系 |
| `屏幕录制 2026-09-29 200003.mp4` | 未打转向灯、连续变道 | 白色目标车锁定 310/314 帧 | 未打灯仍需专门的转向灯时序模型 |

机器可读版本见 `config/china_video_catalog.json`。

两段既有 CCD 预览 `C_000002.mp4`、`C_000008.mp4` 也已纳入统一验证清单。它们来自混合地区事故数据，不用于证明中国道路覆盖，而用于检查车辆检测与跟踪是否依赖前四段屏幕录制的画面风格。

## 新增开放许可外部验证集

以下视频已下载到 `datasets/online_eval/wikimedia`，并校验 SHA-256 与可解码帧数。它们不参加模型训练，但已经用于比较跟踪参数，因此归为 `external_validation`，不能再作为最终盲测集：

| ID | 内容 | 许可 | 测试用途 |
|---|---|---|---|
| `commons_chinese_signal_cc0` | 中国城市路口、红黄灯、车辆和非机动车 | CC0 | 交通灯状态、停止线场景、闯红灯负样本 |
| `commons_china_zebra_collision_pd` | 中国行车记录仪斑马线冲突事故 | Public domain | 行人与车辆冲突、未礼让候选、事故上下文 |

另外登记了两条公共领域逆行视频：上海高架出口掉头逆行，以及三次逆向行驶后碰撞。下载镜像在本次检索时严重限速，因此只登记远程来源，没有把不完整文件留在数据目录。这两条保持 `holdout_candidate`，下载后先运行一次盲测，再决定是否转为验证数据。

缺少这两条视频不会影响检测、跟踪或规则分析器运行，只会减少逆行规则的独立盲测覆盖。替代源复核后没有为了数量降低准入标准：PaddleDetection 的 PP-Vehicle 逆行模块是 Apache-2.0 的有效算法参考，但其演示 GIF 只有 0.96 秒、仅 32 帧且已经叠加车道线和检测框；一个包含 19.5 MB 逆行视频的 GitHub 仓库没有许可证且道路地区未核实；另外两个 MIT 项目不包含视频。因此它们均未加入训练集或盲测集，现有 `195839` 继续只作开发验证。

网络恢复后可按目录 ID 续传下载。工具只在视频能够解码且目录中的 SHA-256（若有）匹配后才把 `.part` 文件原子改为正式文件，同时保存来源与许可侧车清单：

```powershell
python scripts/download_catalog_video.py --id commons_shanghai_elevated_wrong_way_pd
python scripts/download_catalog_video.py --id commons_china_repeated_wrong_way_pd
```

下载不会改变 `holdout_candidate` 身份，也不会自动允许训练。两条视频必须先各运行一次盲测并冻结原始输出，之后才能查看结果或调整参数。

单独验证新增视频：

```powershell
python scripts/run_china_validation_suite.py `
  --output-dir recordings/online_validation `
  --id commons_chinese_signal_cc0 `
  --id commons_china_zebra_collision_pd `
  --device 0 --scene-context
```

批量脚本默认仍使用 Ultralytics 的 `bytetrack.yaml`。候选配置 `config/bytetrack_traffic.yaml` 在两条外部验证视频上减少了短轨迹，但尚未经过新的盲测，因此只能通过 `--tracker config/bytetrack_traffic.yaml` 显式启用，暂不替代默认值。结果会统计轨迹中位持续帧数、短轨迹数量和至少持续 10 帧的稳定轨迹。

本轮 A/B 结果：路口视频车辆轨迹从 24 降至 16；汽车轨迹中位持续帧从 18 提高至 29。斑马线视频车辆轨迹从 88 降至 55；汽车中位持续帧从 12 提高至 27.5，并保留约 96% 汽车观测。这个结果只能说明候选参数值得盲测，不能单独证明 ID 切换率已经改善。

车辆子类别平滑也已经在默认 `bytetrack.yaml` 上完成回归。路口视频仍保留 543 次汽车、173 次摩托车、1,771 次行人和 1,075 次交通灯观测；斑马线视频仍保留 396 次行人观测，并把一个轨迹内的 `car/truck` 抖动从 2 条降至 1 条。剩余切换都是 `vehicle/person` 跨类别冲突，平滑器会立即清空历史而不强行覆盖当前检测，以免用旧车辆标签掩盖跟踪 ID 错配。完整结果见 `recordings/online_validation_smoothed/china_validation_suite.json`。

## 官方测试候选

以下页面来自央视，可作为人工复核和功能测试的素材线索。没有找到允许重新分发或训练的明确许可，因此不自动下载为训练集。

- 闯红灯：央视《热线12》“上海：三辆大货车闯红灯 行车记录仪记下惊险一幕”，69 秒
  <https://tv.cctv.com/2017/10/10/VIDEP501nkVhE7Vejcw6sXkH171010.shtml>
- 车窗抛物：央视“武汉市民视频举报车窗抛物”，152 秒
  <https://tv.cctv.com/2012/12/17/VIDE1355718952149887.shtml>

## 中国道路公开数据

### D2-City

- 论文：<https://arxiv.org/abs/1904.01975>
- 超过 10,000 段来自中国真实交通场景的行车视频。
- 提供车辆检测和跟踪标注，适合车辆检测、跟踪和中国道路域适配。
- 不提供本项目所需的交通违法行为标签，不能直接训练闯红灯、未礼让或车窗抛物规则。
- 下载入口和数据使用条款需要在正式下载前单独核实。

### ApolloScape

- 官方工具仓库：<https://github.com/ApolloScapeAuto/dataset-api>
- 适合车道分割、道路几何、场景解析、3D 跟踪和轨迹任务。
- GitHub 工具代码采用 Apache-2.0；原始数据仍需遵守数据页面自己的条款。
- 不是交通违法事件数据集。

### DAIR-V2X / V2X-Seq

- 官方代码和说明：<https://github.com/AIR-THU/DAIR-V2X>
- 真实中国道路的车端/路侧数据，包含连续场景、轨迹和交通灯相关信息，适合补充车辆跟踪、交通灯上下文和轨迹模型。
- 不提供本项目七类违法行为的直接标签；仓库代码的 Apache-2.0 许可不等于原始数据自动可训练，数据条款仍需单独核实。

### CULane

- 项目页：<https://xingangpan.github.io/projects/CULane.html>
- 北京道路车道线数据，适合车道线/道路几何和实线场景增强。
- 没有交通违法标签；在下载或训练前核实官方数据许可。

### DADA-2000

- 项目仓库：<https://github.com/JWFangit/LOTVS-DADA>
- 大规模驾驶事故和注意力数据，可作为事故/夜间/困难样本的异常检测补充。
- 不是中国道路专用数据，也不等同于闯红灯、压实线或不打灯标签；下载量大，当前不自动下载。

### DoTA

- 项目仓库：<https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly>
- 提供交通异常时序标签、目标轨迹和光流特征；项目说明曾提到获得分享已下载片段的授权。
- 它不是中国道路专用数据，且授权状态应以当前项目条款为准；现有本地目录只有标注，没有视频归档。

### 继续扩充的公开视频数据

- S2TLD：5,786 张道路图像、14,130 个交通灯实例，含红、黄、绿、熄灭和待行状态；适合小目标交通灯感知。
- CCD：1,500 个事故短视频和 3,000 个正常短视频，含昼夜、雨雪和事故参与者标签；适合异常与困难负样本。
- BDD100K：100,000 段多天气、多时段驾驶视频；适合车辆、车道和跟踪的跨场景泛化。
- UA-DETRAC：北京、天津的 100 段交通监控视频；适合不同机位、遮挡和天气下的车辆检测/跟踪。
- Rope3D：50,000 张中国路侧图像和超过 150 万个 3D 目标；适合机位和天气变化，但不是视频违法数据。

这些项目的代码许可不必然覆盖数据本身。必须在下载页再次确认数据条款，尤其是来自 YouTube 或其他二次来源的视频。

## 防止只记住训练视频

1. 按原始长视频、上传者和场景分组，同一组不得跨训练、验证和保留测试集。
2. `training` 用于学习；`validation` 只调阈值；`holdout` 在最终报告前不得训练或调参。
3. 每种行为同时保留正样本和相似负样本，例如红灯停车、合法变道、行人已通过后车辆起步。
4. 分别报告白天、夜间、雨天、远距离、小目标和不同摄像机角度的召回率与误报率。
5. 违法判断仍由“目标检测 + 时序跟踪 + 道路几何 + 规则证据”组成，不能靠单帧分类记忆视频背景。

## 批量验证

先只查看可用素材：

```powershell
python scripts/run_china_validation_suite.py --output-dir D:/桌面/dangerous_driving_test/china_validation_results --dry-run
```

在 Windows 电脑上有 `models/yolov8n.pt` 后运行车辆跟踪基线：

```powershell
python scripts/run_china_validation_suite.py --output-dir D:/桌面/dangerous_driving_test/china_validation_results --device 0 --scene-context
```

输出的跟踪视频和 JSON 仍然只是研发证据；需要道路几何、信号灯、转向灯、行人和抛出物观测后，规则分析器才会生成对应的“疑似”事件。

## 真实规则管线复核

- `195716` 的压实线候选经逐帧视觉复核，红框 `track_id=4` 对应画面中的压线 SUV；该结果仍只标记为疑似违法并等待人工复核。
- `195920` 的斑马线候选加入保守事件合并后，从 4 条提示降为 2 条。同一车辆 `track_id=3`、同一斑马线在 1 秒内产生的三个候选被合并，证据保留行人轨迹 ID `22/48/66`，避免行人 ID 抖动造成重复报警。
- 后段车辆 `track_id=84` 没有与 `track_id=3` 自动合并。它们可能是同一物理车辆的跟踪 ID 切换，但当前证据不足以可靠重关联；为避免锁错违法车辆，继续保留为独立的低置信度人工复核事件。
- 新版斑马线复核视频为 `D:\桌面\dangerous_driving_test\offense_validation_current\195920_crosswalk_review_v2.mp4`。

## 下一步优先级

1. 用 `200003` 完成目标锁定和连续变道轨迹验证。
2. 用 `195920` 标定斑马线并组合车辆、行人与速度证据。
3. 为 `195716` 标定实线；若字幕遮挡关键区域，则替换为央视或原始无字幕视频。
4. 为 `195839` 增加夜间增强并重新评估召回率。
5. 从官方候选补充闯红灯与车窗抛物测试，但训练前必须取得可用许可或自采数据。

任何自动结果都只能标记为“疑似违法、待人工复核”，不能直接作为行政执法结论。
