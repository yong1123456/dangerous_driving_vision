#!/usr/bin/env python3
"""Safe Yahboom servo-controller protocol adapter for the 2DOF PTZ.

The packet layout follows the public Yahboom 2DOF-PTZ examples.  This module
does not open a serial port or move hardware unless the caller explicitly
creates a transport and sends a command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class ByteTransport(Protocol):
    def write(self, data: bytes) -> int: ...


HEAD = 0xFF
DEFAULT_DEVICE_ID = 0xFC
FUNC_PWM_SERVO = 0x03
FUNC_PWM_SERVO_ALL = 0x04


def _byte(value: int, name: str) -> int:
    value = int(value)
    if not 0 <= value <= 0xFF:
        raise ValueError(f"{name} must be an unsigned byte")
    return value


def build_packet(
    function: int,
    payload: Sequence[int] = (),
    *,
    device_id: int = DEFAULT_DEVICE_ID,
) -> bytes:
    """Build a Yahboom frame: FF, device, length, function, payload, checksum."""
    device_id = _byte(device_id, "device_id")
    function = _byte(function, "function")
    payload_bytes = [_byte(value, "payload value") for value in payload]
    # The vendor examples set length to len(frame_without_checksum) - 1:
    # device id + length byte + function + payload.
    length = 3 + len(payload_bytes)
    if length > 0xFF:
        raise ValueError("payload is too long")
    packet = [HEAD, device_id, length, function, *payload_bytes]
    checksum = (sum(packet) + (257 - device_id)) & 0xFF
    packet.append(checksum)
    return bytes(packet)


def build_servo_packet(
    channel: int,
    angle: int,
    *,
    device_id: int = DEFAULT_DEVICE_ID,
    max_channel: int = 24,
) -> bytes:
    """Build a single PWM-servo command for a controller channel."""
    if not 1 <= int(channel) <= max_channel:
        raise ValueError(f"channel must be between 1 and {max_channel}")
    if not 0 <= int(angle) <= 180:
        raise ValueError("controller servo angle must be between 0 and 180")
    return build_packet(FUNC_PWM_SERVO, (int(channel), int(angle)), device_id=device_id)


def build_servo_all_packet(
    angles: Sequence[int | None], *, device_id: int = DEFAULT_DEVICE_ID
) -> bytes:
    """Build the four-channel command; None leaves a channel unchanged."""
    if len(angles) != 4:
        raise ValueError("the all-servo command requires exactly four values")
    values: list[int] = []
    for angle in angles:
        if angle is None:
            values.append(255)
        elif 0 <= int(angle) <= 180:
            values.append(int(angle))
        else:
            raise ValueError("servo angles must be 0..180 or None")
    return build_packet(FUNC_PWM_SERVO_ALL, values, device_id=device_id)


@dataclass(frozen=True)
class PtzChannels:
    pan_channel: int = 1
    tilt_channel: int = 2
    device_id: int = DEFAULT_DEVICE_ID


class YahboomPtz:
    """Explicit-command wrapper.  No command is sent during construction."""

    def __init__(self, transport: ByteTransport, channels: PtzChannels | None = None):
        self.transport = transport
        self.channels = channels or PtzChannels()

    def set_channel_angle(self, channel: int, angle: int) -> bytes:
        packet = build_servo_packet(
            channel,
            angle,
            device_id=self.channels.device_id,
        )
        self.transport.write(packet)
        return packet

    def set_pan_tilt(self, pan_angle: int, tilt_angle: int) -> tuple[bytes, bytes]:
        pan = self.set_channel_angle(self.channels.pan_channel, pan_angle)
        tilt = self.set_channel_angle(self.channels.tilt_channel, tilt_angle)
        return pan, tilt


@dataclass(frozen=True)
class AimStep:
    pan_delta: float
    tilt_delta: float


def compute_aim_step(
    error_x: float,
    error_y: float,
    *,
    frame_width: int,
    frame_height: int,
    gain: float = 8.0,
    deadband: float = 0.06,
    max_step: float = 4.0,
) -> AimStep:
    """Convert normalized target-center error into a bounded PTZ step.

    Positive x moves pan right and positive y moves tilt down.  Sign and
    channel orientation must be verified with the actual mechanical mounting.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if gain <= 0 or deadband < 0 or max_step <= 0:
        raise ValueError("gain/max_step must be positive and deadband non-negative")
    normalized_x = float(error_x) / frame_width
    normalized_y = float(error_y) / frame_height

    def step(error: float) -> float:
        if abs(error) <= deadband:
            return 0.0
        value = gain * error
        return max(-max_step, min(max_step, value))

    return AimStep(pan_delta=step(normalized_x), tilt_delta=step(normalized_y))
