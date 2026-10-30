#!/usr/bin/env python3
"""
imu_check.py -- standalone check that an IMU on the Jetson's I2C header works,
and how good its heading is. No other project files needed.

Wiring (Xavier NX 40-pin header, I2C bus 8):
    VCC -> pin 1 (3.3 V, NOT 5 V)   GND -> pin 6
    SDA -> pin 3                    SCL -> pin 5

Auto-detects, on every /dev/i2c-* bus:
    MPU6050 / MPU6500 / MPU9250 / MPU9255   address 0x68 or 0x69
    BNO055                                   address 0x28 or 0x29

Tests:
    1. detection     which chip, which bus, which address
    2. at rest       gravity reads ~9.81 m/s^2; which axis points down;
                     gyro bias and noise; achievable sample rate
    3. drift         keep still 20 s; integrate the yaw rate and report how
                     fast heading drifts -- the number that limits how far the
                     UGV can dead-reckon before it leaves its lane
    4. rotation      (optional, --rotate) turn the board 90 deg flat on the
                     table; the integrated angle should read ~90

Usage:
    python3 imu_check.py              # tests 1-3
    python3 imu_check.py --rotate     # also test 4
    python3 imu_check.py --live       # just stream readings (Ctrl-C to stop)

Needs: sudo apt install python3-smbus   (and the user in the 'i2c' group)
"""

import argparse
import glob
import math
import re
import statistics
import struct
import sys
import time

try:
    from smbus import SMBus
except ImportError:
    try:
        from smbus2 import SMBus
    except ImportError:
        sys.exit("smbus missing -- run: sudo apt install python3-smbus")

G = 9.80665
RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append(ok)
    print("  [{}] {:<40} {}".format("PASS" if ok else "FAIL", name, detail))


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------

class MPU:
    """InvenSense MPU6050 / 6500 / 9250 family (accel + gyro).
    No magnetometer used, so heading comes only from integrating the gyro --
    it WILL drift; test 3 measures how fast."""
    WHO = {0x68: "MPU6050", 0x70: "MPU6500", 0x71: "MPU9250", 0x73: "MPU9255"}

    def __init__(self, bus, addr, who):
        self.bus, self.addr = bus, addr
        self.name = self.WHO.get(who, "MPU-compatible (WHO_AM_I=0x{:02X})".format(who))
        b = self.bus
        b.write_byte_data(addr, 0x6B, 0x01)   # wake, clock from X gyro PLL
        time.sleep(0.1)
        b.write_byte_data(addr, 0x1A, 0x03)   # low-pass filter ~44 Hz: less noise
        b.write_byte_data(addr, 0x1B, 0x00)   # gyro +-250 dps  -> 131 LSB per dps
        b.write_byte_data(addr, 0x1C, 0x00)   # accel +-2 g     -> 16384 LSB per g
        time.sleep(0.05)
        self.temp_scale, self.temp_off = ((340.0, 36.53) if who == 0x68 else (333.87, 21.0))

    def read(self):
        raw = bytes(self.bus.read_i2c_block_data(self.addr, 0x3B, 14))
        ax, ay, az, t, gx, gy, gz = struct.unpack(">7h", raw)    # big-endian int16
        acc = [v / 16384.0 * G for v in (ax, ay, az)]            # m/s^2
        gyr = [v / 131.0 for v in (gx, gy, gz)]                  # deg/s
        return acc, gyr, t / self.temp_scale + self.temp_off, None


class BNO055:
    """Bosch BNO055: accel + gyro + magnetometer with on-chip fusion, so it
    reports an absolute heading. The compass is easily disturbed by motors
    and steel -- mount it away from them."""
    name = "BNO055"

    def __init__(self, bus, addr):
        self.bus, self.addr = bus, addr
        b = self.bus
        b.write_byte_data(addr, 0x3D, 0x00)   # CONFIG mode
        time.sleep(0.03)
        b.write_byte_data(addr, 0x3E, 0x00)   # normal power
        b.write_byte_data(addr, 0x3B, 0x00)   # units: m/s^2, deg/s, degrees, C
        b.write_byte_data(addr, 0x3D, 0x0C)   # NDOF fusion mode
        time.sleep(0.05)

    def read(self):
        acc = struct.unpack("<3h", bytes(self.bus.read_i2c_block_data(self.addr, 0x08, 6)))
        gyr = struct.unpack("<3h", bytes(self.bus.read_i2c_block_data(self.addr, 0x14, 6)))
        eul = struct.unpack("<3h", bytes(self.bus.read_i2c_block_data(self.addr, 0x1A, 6)))
        temp = struct.unpack("b", bytes([self.bus.read_byte_data(self.addr, 0x34)]))[0]
        return ([v / 100.0 for v in acc],          # m/s^2
                [v / 16.0 for v in gyr],           # deg/s
                float(temp),
                [v / 16.0 for v in eul])           # heading, roll, pitch (deg)

    def calibration(self):
        c = self.bus.read_byte_data(self.addr, 0x35)
        return {"sys": c >> 6 & 3, "gyro": c >> 4 & 3, "accel": c >> 2 & 3, "mag": c & 3}


def detect():
    buses = sorted(int(re.findall(r"\d+", p)[-1]) for p in glob.glob("/dev/i2c-*"))
    if not buses:
        sys.exit("No /dev/i2c-* buses found.")
    order = [b for b in (8, 1) if b in buses] + [b for b in buses if b not in (8, 1)]
    for busno in order:
        try:
            bus = SMBus(busno)
        except (OSError, PermissionError) as e:
            print("  bus {}: cannot open ({}) -- in the i2c group?".format(busno, e))
            continue
        for addr in (0x68, 0x69):
            try:
                who = bus.read_byte_data(addr, 0x75)
                return busno, addr, MPU(bus, addr, who)
            except OSError:
                pass
        for addr in (0x28, 0x29):
            try:
                if bus.read_byte_data(addr, 0x00) == 0xA0:
                    return busno, addr, BNO055(bus, addr)
            except OSError:
                pass
        bus.close()
    return None, None, None


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def sample(imu, seconds):
    out, t0 = [], time.time()
    while time.time() - t0 < seconds:
        out.append((time.time(), imu.read()))
    return out


def test_rest(imu):
    print("\n=== 2. at rest (keep it still for 3 s) ===")
    time.sleep(0.5)
    data = sample(imu, 3.0)
    rate = len(data) / max(data[-1][0] - data[0][0], 1e-6)
    acc = [d[1][0] for d in data]
    gyr = [d[1][1] for d in data]
    mag = [math.sqrt(sum(c * c for c in a)) for a in acc]
    mean_g = statistics.mean(mag)
    record("gravity magnitude {:.2f} m/s^2".format(mean_g),
           abs(mean_g - G) < 0.5, "expect 9.81 +- 0.5")
    means = [statistics.mean(a[i] for a in acc) for i in range(3)]
    down = max(range(3), key=lambda i: abs(means[i]))
    sign = "+" if means[down] > 0 else "-"
    print("        gravity is on the {}{} axis (accel x/y/z = {:+.2f} {:+.2f} {:+.2f})".format(
        sign, "XYZ"[down], *means))
    bias = [statistics.mean(g[i] for g in gyr) for i in range(3)]
    noise = [statistics.pstdev(g[i] for g in gyr) for i in range(3)]
    record("gyro bias x/y/z {:+.2f} {:+.2f} {:+.2f} deg/s".format(*bias),
           max(abs(b) for b in bias) < 20.0, "datasheet allows up to +-20")
    print("        gyro noise x/y/z {:.3f} {:.3f} {:.3f} deg/s".format(*noise))
    record("sample rate {:.0f} Hz".format(rate), rate >= 50.0, "control needs >= 50 Hz")
    if data[-1][1][2] is not None:
        print("        temperature {:.1f} C".format(data[-1][1][2]))
    return bias, down


def test_drift(imu, bias, yaw_axis, seconds=20.0):
    print("\n=== 3. heading drift (keep it still for {:.0f} s) ===".format(seconds))
    angle, last, t_end = 0.0, time.time(), time.time() + seconds
    first_eul = None
    while time.time() < t_end:
        acc, gyr, _, eul = imu.read()
        now = time.time()
        angle += (gyr[yaw_axis] - bias[yaw_axis]) * (now - last)
        last = now
        if eul is not None and first_eul is None:
            first_eul = eul[0]
    rate = angle / seconds * 60.0                       # deg per minute
    lateral = 15.0 * math.sin(math.radians(abs(rate)))  # after 1 min, over 15 m
    record("gyro-integrated drift {:+.2f} deg/min".format(rate), abs(rate) < 5.0,
           "bias-corrected, about the {} axis".format("XYZ"[yaw_axis]))
    print("        a heading error that size puts the UGV {:.2f} m sideways after".format(lateral))
    print("        15 m of travel -- compare with the ~0.3 m lane margin")
    if first_eul is not None:
        eul = imu.read()[3]
        d = (eul[0] - first_eul + 180.0) % 360.0 - 180.0
        print("        BNO055 fused heading changed {:+.2f} deg over {:.0f} s; calibration {}".format(
            d, seconds, imu.calibration()))
        print("        (calibration 0 = uncalibrated, 3 = fully; move it in a figure-8 to calibrate mag)")


def test_rotation(imu, bias, yaw_axis):
    print("\n=== 4. rotation ===")
    input("  Place it flat. Press Enter, then turn it 90 deg (quarter turn) within 5 s... ")
    angle, last, t_end = 0.0, time.time(), time.time() + 5.0
    while time.time() < t_end:
        gyr = imu.read()[1]
        now = time.time()
        angle += (gyr[yaw_axis] - bias[yaw_axis]) * (now - last)
        last = now
    record("integrated rotation {:+.1f} deg".format(angle), abs(abs(angle) - 90.0) < 8.0,
           "expect +-90 (sign shows the direction)")


def live(imu):
    print("streaming at 10 Hz, Ctrl-C to stop")
    try:
        while True:
            acc, gyr, t, eul = imu.read()
            line = "acc {:+6.2f} {:+6.2f} {:+6.2f} m/s2 | gyro {:+7.2f} {:+7.2f} {:+7.2f} deg/s".format(*acc, *gyr)
            if eul is not None:
                line += " | heading {:6.1f}".format(eul[0])
            print(line)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotate", action="store_true", help="also run the rotation test")
    ap.add_argument("--live", action="store_true", help="stream readings only")
    a = ap.parse_args()

    print("=== 1. detection ===")
    busno, addr, imu = detect()
    if imu is None:
        record("IMU found on any I2C bus", False,
               "check wiring (SDA pin 3, SCL pin 5, VCC pin 1, GND pin 6) and "
               "run: sudo i2cdetect -y -r 8")
        return 1
    record("found {} on bus {} at 0x{:02X}".format(imu.name, busno, addr), True)

    if a.live:
        live(imu)
        return 0
    bias, down = test_rest(imu)
    test_drift(imu, bias, down)
    if a.rotate:
        test_rotation(imu, bias, down)
    print("\n{}/{} checks passed".format(sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
