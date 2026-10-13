# Traffic violation reference sources

These files are preserved as implementation references. They are not imported
by the application. Each source keeps its upstream license and README.

| Directory | Upstream | License | Relevant implementation |
|---|---|---|---|
| `live_traffic_mit` | `OdigweIfe/Live-Traffic-Detection-project` | MIT | Traffic-light color ROI, stop-line transition, lane ROI configuration |
| `red_light_bsd` | `kunal-gandvane/Red_Light_Violation_Detection` | BSD-3-Clause | Tracked red-light violation workflow and evidence clips |
| `wrong_way_mit` | `Umraz-Hussain-MyWorld/Smart-Vehicle-WrongWay-Detection-And-Tracking` | MIT | Sustained trajectory-direction check and evidence snapshots |
| `paddledetection_ppvehicle_apache2` | `PaddlePaddle/PaddleDetection` | Apache-2.0 | Official lane pressing, wrong-way trajectory logic, lane-label conversion and configuration |

The project implementation remains independent and adds calibrated road
geometry, confidence thresholds, temporal evidence, and human-review notices.
The upstream demo videos are not copied because they are not verified Chinese
road footage and are not needed to use the algorithms as references.

The exact reviewed commits and component limitations are recorded in
`config/open_source_components.json`.
