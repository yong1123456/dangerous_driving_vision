# Open-source traffic-violation references

Last reviewed: 2026-10-13

Only repositories with inspectable implementation files are treated as code
references. A repository description, README claim, or external download link
does not count as an implemented or locally available capability.

## Accepted code references

| Project | License | Verified contents | Use in this project |
|---|---|---|---|
| [Live-Traffic-Detection-project](https://github.com/OdigweIfe/Live-Traffic-Detection-project) | MIT | Red-light state/stop-line logic, lane ROI, tracker pipeline and tests | HSV traffic-light state idea was independently implemented with confidence and coverage checks |
| [Red_Light_Violation_Detection](https://github.com/kunal-gandvane/Red_Light_Violation_Detection) | BSD-3-Clause | YOLOv8, DeepSORT, violation zone, evidence clip logging, four input and four result videos | Used as a workflow reference; demo videos were not copied because their region is not verified as China |
| [Smart-Vehicle-WrongWay-Detection-And-Tracking](https://github.com/Umraz-Hussain-MyWorld/Smart-Vehicle-WrongWay-Detection-And-Tracking) | MIT | YOLOv8, SORT, sustained centroid direction check and evidence snapshots | Compared against the existing direction-alignment rule; existing calibrated rule remains stricter |
| [PaddleDetection PP-Vehicle retrograde](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/deploy/pipeline/docs/tutorials/ppvehicle_retrograde.md) | Apache-2.0 | Official lane-aware wrong-way pipeline | Algorithm reference; its 0.96-second annotated GIF is not accepted as a test video |

The reviewed PP-Vehicle source is now preserved locally at commit
`b25522a0f4bde8c80603f3ba5e3472059972e3b5`. It includes both
`vehicle_pressing.py` and `vehicle_retrograde.py`, the official violation
configuration, lane-mask conversion utility and the upstream license. This is
the strongest complete open-source basis found for solid-line and wrong-way
analysis.

## Additional verified components

| Project | License | Suitable contribution | Important limitation |
|---|---|---|---|
| [Ultra-Fast-Lane-Detection-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) | MIT | Lane geometry model | No offense rules and no ready RKNN artifact |
| [YOLOP](https://github.com/hustvl/YOLOP) | MIT | Joint vehicle, drivable-area and lane perception | Older training distribution; must be tested on Chinese dashcam footage |
| [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) | Apache-2.0 | Vehicle, pedestrian and traffic-light models | Intel/OpenVINO deployment, not direct RK3588 deployment |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) | MIT | Stable multi-object tracking | Tracking alone cannot establish an offense |

Exact commit IDs and integration status are machine-readable in
`config/open_source_components.json`.

## Coverage reality

| Requested behavior | Best reusable basis | Current gap |
|---|---|---|
| Solid-line crossing | PP-Vehicle lane segmentation plus the local calibrated crossing rule | Convert or train a lane model for RKNN and validate it on Chinese roads |
| Wrong-way driving | PP-Vehicle and the local sustained-direction rule | Calibrated legal direction for each lane and ego-motion compensation |
| Red-light running | Local light-state classifier plus accepted red-light workflow | Associate each signal with a lane and observe a before-to-after stop-line transition |
| Consecutive lane changes | Local temporal rule plus a lane model | Reliable per-frame lane assignment while the vehicle and PTZ are moving |
| Lane change without signal | No licensed complete implementation was found | A rear turn-lamp temporal classifier and Chinese-road validation data are required |
| Failure to yield to pedestrians | Local crosswalk interaction rule | Crosswalk calibration, person tracking and right-of-way context are required |
| Littering from a vehicle | No licensed complete implementation was found | A litter-object detector, source-vehicle association and dedicated labeled clips are required |

The last two gaps cannot be solved by copying the small repositories returned
by search. The turn-signal candidate includes a model but no license. The
littering candidates contain only five or six files, have no reusable license,
and do not provide a validated dataset/model pair.

Preserved source files and their upstream licenses are under
`third_party/traffic_violation_references`. They are references and are not
imported by the runtime.

## Rejected or limited candidates

| Candidate | Decision |
|---|---|
| `TJW-XMUT/QT-Red-Light-Violation-Wrong-Way-Driving-Detection-Traffic-Counting-License-Plate-Recognition` | Rejected as implementation source: the repository contains only `README.md` and `LICENSE`, despite claiming code, models and RK3588 support |
| `Thinklab-SJTU/S2TLD` | Useful traffic-light dataset lead, but the repository contains only README/license and external dataset links; dataset terms still require separate verification |
| `cdmstrong/yolov5-deepSort-tracker` | GPL code is present, but it is an older general counter/tracker and does not add a stronger state model than the accepted references |
| Unlicensed turn-signal repositories | Not copied or integrated; repository availability is not a reuse license |
| `qiekuo/Visual-recognition-AI` | Rejected: only 16 files are present, the core packages described in its README are absent, and it has no license |
| Small vehicle-littering repositories | Rejected for integration: no explicit license and no complete validated model/dataset pair |
| GitHub demo videos from non-China or unverified regions | Not counted as Chinese-road validation data |

## Implemented result

`src/traffic_light_state.py` now classifies a detected traffic-light box as
`red`, `yellow`, `green`, or `unknown`. The tracker stores the state,
state confidence, and per-color evidence in its JSON output.

Validation on the downloaded China intersection video produced 1,075 light
observations: 385 red, 88 yellow, 190 green, and 412 unknown. Visual review
confirmed that the principal red and green signals are labelled correctly.
This is traffic-light perception validation, not evidence that a vehicle ran a
red light; a calibrated stop line and a tracked crossing are still required.

All automatic offense results remain suspected events for human review and
are not legal findings.
