import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yahboom_ptz import (  # noqa: E402
    FUNC_PWM_SERVO,
    PtzChannels,
    YahboomPtz,
    build_packet,
    build_servo_all_packet,
    build_servo_packet,
    compute_aim_step,
)


class FakeTransport:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)


class YahboomPtzTest(unittest.TestCase):
    def test_servo_packet_matches_public_protocol_layout(self) -> None:
        packet = build_servo_packet(1, 90)
        self.assertEqual(packet[:6], bytes([0xFF, 0xFC, 0x05, FUNC_PWM_SERVO, 1, 90]))
        self.assertEqual(packet[-1], 99)

    def test_all_servo_packet_uses_255_for_unchanged_channels(self) -> None:
        packet = build_servo_all_packet((90, None, 30, None))
        self.assertEqual(packet[4:8], bytes([90, 255, 30, 255]))

    def test_wrapper_sends_only_explicit_commands(self) -> None:
        transport = FakeTransport()
        ptz = YahboomPtz(transport, PtzChannels(pan_channel=3, tilt_channel=4))
        self.assertEqual(transport.writes, [])
        pan, tilt = ptz.set_pan_tilt(90, 80)
        self.assertEqual(transport.writes, [pan, tilt])

    def test_aim_step_has_deadband_and_limit(self) -> None:
        self.assertEqual(
            compute_aim_step(0, 0, frame_width=1920, frame_height=1080),
            type(compute_aim_step(0, 0, frame_width=1920, frame_height=1080))(0.0, 0.0),
        )
        step = compute_aim_step(1920, -1080, frame_width=1920, frame_height=1080)
        self.assertEqual(step.pan_delta, 4.0)
        self.assertEqual(step.tilt_delta, -4.0)


if __name__ == "__main__":
    unittest.main()
