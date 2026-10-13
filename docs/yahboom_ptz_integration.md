# Yahboom 2DOF PTZ integration

The public Yahboom 2DOF-PTZ repository is useful as a hardware reference:

`https://github.com/YahboomTechnology/2DOF-PTZ/tree/main/3.K230%20module%20on%20PTZ/7.%20Moving%20target%20control%20and%20tracking`

The K210 example exposes the controller frame used by the servo board:

```text
FF FC LENGTH FUNCTION PAYLOAD... CHECKSUM
```

For a single PWM servo, function `0x03` is used. The current adapter is in
`src/yahboom_ptz.py`; it implements packet construction, explicit channel
commands, and a bounded target-centering step. It does not open a serial port
or move the gimbal by itself.

## Hardware interpretation

The pictured controller has a 6.5-8.4 V power input, a 24-channel servo
interface, a USB Type-C connector, and a serial-port-3 connector. The two PTZ
servos should be powered from the controller's servo supply. Do not power the
high-torque servos from Orange Pi 5 Pro GPIO or USB.

The Orange Pi should communicate with the controller through the controller's
documented USB/serial interface. The actual Linux device (`/dev/ttyUSB*` or
`/dev/ttyACM*`), baud rate, pan channel, tilt channel, and direction must be
verified on the physical board before enabling motion. The repository examples
do not establish these four values for every controller revision.

## Software control loop

1. RKNN detects and ByteTrack tracks a vehicle.
2. The target-lock layer confirms one target and rejects ambiguous switches.
3. `compute_aim_step()` converts the target-center error to a small bounded pan/tilt step.
4. A hardware-specific serial transport calls `YahboomPtz.set_pan_tilt()` at a limited rate.
5. Road-offense geometry is disabled while the camera is moving, then resumes only after the calibrated view is stable.

Before attaching the camera, run the adapter with a fake transport and log the
packets. On the real board, begin with small steps and a physical emergency
power switch. Mechanical limits and sign conventions must be calibrated by
hand.
