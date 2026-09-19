#!/usr/bin/env python3
"""Continuous camera recorder with a rolling buffer and manual event trigger."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "recorder.json"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = {
        "camera_device",
        "input_format",
        "width",
        "height",
        "fps",
        "recorder_backend",
        "video_bitrate",
        "gop",
        "segment_seconds",
        "buffer_seconds",
        "pre_event_seconds",
        "post_event_seconds",
        "ffmpeg",
        "gstreamer",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")
    return config


class RollingRecorder:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.buffer_dir = PROJECT_ROOT / "recordings" / "buffer"
        self.events_dir = PROJECT_ROOT / "recordings" / "events"
        self.logs_dir = PROJECT_ROOT / "logs"
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle: Any = None
        self.stop_event = threading.Event()
        self.cleanup_thread: threading.Thread | None = None
        self.event_threads: list[threading.Thread] = []

    def start(self) -> None:
        device = Path(str(self.config["camera_device"]))
        if not device.exists():
            raise RuntimeError(f"Camera device does not exist: {device}")

        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        for old_segment in self.buffer_dir.glob("*.mkv"):
            old_segment.unlink()

        command = self._capture_command(device)
        log_path = self.logs_dir / "capture-recorder.log"
        self.log_handle = log_path.open("ab", buffering=0)
        self.process = subprocess.Popen(
            command,
            stdout=self.log_handle,
            stderr=self.log_handle,
        )

        time.sleep(2)
        if self.process.poll() is not None:
            raise RuntimeError(f"FFmpeg stopped during startup. Check {log_path}")

        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="buffer-cleanup",
            daemon=True,
        )
        self.cleanup_thread.start()

    def _capture_command(self, device: Path) -> list[str]:
        backend = str(self.config["recorder_backend"])
        if backend != "gstreamer_mpp":
            raise ValueError(f"Unsupported recorder backend: {backend}")

        executable = str(self.config["gstreamer"])
        if shutil.which(executable) is None:
            raise RuntimeError(f"GStreamer executable not found: {executable}")

        segment_nanoseconds = int(float(self.config["segment_seconds"]) * 1_000_000_000)
        output_pattern = self.buffer_dir / "segment_%05d.mkv"
        caps = (
            f'image/jpeg,width={self.config["width"]},'
            f'height={self.config["height"]},framerate={self.config["fps"]}/1'
        )
        raw_caps = (
            f'video/x-raw,width={self.config["width"]},'
            f'height={self.config["height"]},format=NV12'
        )
        return [
            executable,
            "-e",
            "v4l2src",
            f"device={device}",
            "!",
            caps,
            "!",
            "jpegparse",
            "!",
            "jpegdec",
            "!",
            "videoconvert",
            "!",
            raw_caps,
            "!",
            "mpph264enc",
            f'bps={self.config["video_bitrate"]}',
            "rc-mode=cbr",
            f'gop={self.config["gop"]}',
            "!",
            "h264parse",
            "!",
            "splitmuxsink",
            f"location={output_pattern}",
            f"max-size-time={segment_nanoseconds}",
            "muxer-factory=matroskamux",
        ]

    def trigger(self) -> str:
        event_id = f'{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}'
        trigger_time = time.time()
        thread = threading.Thread(
            target=self._finalize_event,
            args=(event_id, trigger_time),
            name=f"event-{event_id}",
        )
        self.event_threads.append(thread)
        thread.start()
        return event_id

    def stop(self) -> None:
        self.stop_event.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

        for thread in self.event_threads:
            if thread.is_alive():
                thread.join()

        if self.log_handle:
            self.log_handle.close()

    def _cleanup_loop(self) -> None:
        segment_seconds = float(self.config["segment_seconds"])
        minimum_retention = (
            float(self.config["pre_event_seconds"])
            + float(self.config["post_event_seconds"])
            + 2 * segment_seconds
            + 5
        )
        retention = max(float(self.config["buffer_seconds"]), minimum_retention)

        while not self.stop_event.wait(1):
            cutoff = time.time() - retention
            for segment in self.buffer_dir.glob("*.mkv"):
                try:
                    if segment.stat().st_mtime < cutoff:
                        segment.unlink()
                except FileNotFoundError:
                    pass

    def _finalize_event(self, event_id: str, trigger_time: float) -> None:
        post_seconds = float(self.config["post_event_seconds"])
        segment_seconds = float(self.config["segment_seconds"])
        time.sleep(post_seconds + segment_seconds + 1)

        start_time = trigger_time - float(self.config["pre_event_seconds"])
        end_time = trigger_time + post_seconds
        event_dir = self.events_dir / f"event_{event_id}"
        segments_dir = event_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=False)

        selected: list[Path] = []
        for segment in sorted(self.buffer_dir.glob("*.mkv")):
            try:
                modified = segment.stat().st_mtime
            except FileNotFoundError:
                continue
            if start_time - segment_seconds <= modified <= end_time + segment_seconds + 1:
                selected.append(segment)

        copied: list[Path] = []
        for source in selected:
            destination = segments_dir / source.name
            try:
                shutil.copy2(source, destination)
                copied.append(destination)
            except FileNotFoundError:
                continue

        metadata = {
            "event_id": event_id,
            "event_type": "manual_test_trigger",
            "trigger_time_local": datetime.fromtimestamp(trigger_time).astimezone().isoformat(),
            "trigger_unix_seconds": trigger_time,
            "requested_pre_event_seconds": self.config["pre_event_seconds"],
            "requested_post_event_seconds": self.config["post_event_seconds"],
            "camera": {
                "device": self.config["camera_device"],
                "format": self.config["input_format"],
                "width": self.config["width"],
                "height": self.config["height"],
                "fps": self.config["fps"],
            },
            "recording": {
                "backend": self.config["recorder_backend"],
                "codec": "h264",
                "bitrate": self.config["video_bitrate"],
                "gop": self.config["gop"],
            },
            "segment_count": len(copied),
            "status": "complete" if copied else "no_segments",
        }

        if copied:
            concat_list = event_dir / "segments.txt"
            concat_list.write_text(
                "".join(f"file 'segments/{path.name}'\n" for path in copied),
                encoding="utf-8",
            )
            raw_video = event_dir / "raw.mkv"
            concat_command = [
                str(self.config["ffmpeg"]),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy",
                str(raw_video),
            ]
            concat_result = subprocess.run(concat_command, check=False)
            metadata["concat_exit_code"] = concat_result.returncode

            if concat_result.returncode == 0:
                scene_path = event_dir / "scene.jpg"
                scene_command = [
                    str(self.config["ffmpeg"]),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    str(self.config["pre_event_seconds"]),
                    "-i",
                    str(raw_video),
                    "-frames:v",
                    "1",
                    str(scene_path),
                ]
                scene_result = subprocess.run(scene_command, check=False)
                metadata["scene_exit_code"] = scene_result.returncode

        metadata_path = event_dir / "event.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._write_manifest(event_dir)
        print(f"\nEvent saved: {event_dir}", flush=True)

    @staticmethod
    def _write_manifest(event_dir: Path) -> None:
        entries: list[str] = []
        for path in sorted(event_dir.rglob("*")):
            if not path.is_file() or path.name == "manifest.sha256":
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            relative = path.relative_to(event_dir).as_posix()
            entries.append(f"{digest}  {relative}\n")
        (event_dir / "manifest.sha256").write_text("".join(entries), encoding="ascii")


def main() -> int:
    config = load_config(DEFAULT_CONFIG)
    recorder = RollingRecorder(config)
    try:
        recorder.start()
        print("Recorder running. Enter 't' to trigger an event, 'q' to quit.")
        while True:
            command = input("> ").strip().lower()
            if command in {"t", "trigger"}:
                event_id = recorder.trigger()
                print(f"Triggered {event_id}; collecting post-event video...")
            elif command in {"q", "quit", "exit"}:
                break
            elif command:
                print("Unknown command. Use 't' or 'q'.")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        recorder.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
