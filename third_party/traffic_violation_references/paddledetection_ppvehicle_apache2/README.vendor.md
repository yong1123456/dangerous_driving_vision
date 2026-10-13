# PaddleDetection PP-Vehicle reference

Upstream: https://github.com/PaddlePaddle/PaddleDetection

- Branch: `release/2.9`
- Commit: `b25522a0f4bde8c80603f3ba5e3472059972e3b5`
- License: Apache-2.0 (see `LICENSE`)
- Retrieved: 2026-10-13

The preserved files implement or document the official PP-Vehicle vehicle
violation pipeline:

- `vehicle_pressing.py`: associates tracked vehicles with segmented solid
  lane markings.
- `vehicle_retrograde.py`: uses tracked trajectories, lane direction and a
  calibrated fence line to find wrong-way candidates.
- `infer_cfg_vehicle_violation.yml`: official model and rule configuration.
- `lane_to_mask.py`: converts BDD100K lane annotations into segmentation
  masks with background, double-yellow, solid and dashed classes.
- `ppvehicle_violation.md`: official lane-model customization guide.

These files are preserved verbatim for implementation comparison and license
compliance. They are not imported by the current runtime. The upstream
Paddle models are not RKNN files and require a separate export/conversion and
accuracy validation before Orange Pi deployment.
