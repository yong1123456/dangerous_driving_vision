# Vendored Supervision ByteTrack

Source: roboflow/supervision
Version: 0.18.0
Commit: 53f4cde18d353addb54380c237376dfbcf58067d
License: MIT, see LICENSE.md

The files under supervision/tracker/byte_tracker are unmodified upstream files.
The local supervision/detection package is a compatibility shim containing only
the Detections container and box IoU helper required by ByteTrack. This avoids
installing the full Supervision package and upgrading the Orange Pi RKNN
environment.

Runtime dependencies are NumPy and SciPy.
