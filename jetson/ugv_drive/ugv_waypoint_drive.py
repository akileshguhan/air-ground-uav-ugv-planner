#!/usr/bin/env python3
"""
ugv_waypoint_drive.py -- drive the UGV through a list of waypoints using the
MPU6050 for heading and calibrated speed x time for distance.

WHY THIS SPLIT: the IMU measures rotation well but cannot measure distance --
integrating acceleration twice turns a 0.03 m/s^2 error into 1.5 m after 10 s.
So:  heading  <- gyro (integrated, bias-calibrated, spike-filtered)
     distance <- measured speed (m/s) x driving time
Strategy per waypoint: turn in place to face it, then drive straight for
distance / speed seconds while holding that heading.

FRAME: the vehicle starts at (0, 0) facing +x. +y is to its left. Heading is
counter-clockwise positive, viewed from above.

STEPS, IN ORDER (wheels on the ground for 2-4; battery switch within reach):
  1. python3 ugv_waypoint_drive.py --dry-run --demo square
        no hardware: simulates the robot, checks the control logic
  2. python3 ugv_waypoint_drive.py --check-turn
        spins left briefly; confirms motors and IMU agree on direction
  3. python3 ugv_waypoint_drive.py --calibrate-speed 3
        drives straight 3 s; you measure the distance; saves m/s
  4. python3 ugv_waypoint_drive.py --demo square --scale 0.5
        drives a 0.5 m square
     python3 ugv_waypoint_drive.py --waypoints path_msg.json --scale 0.1
        drives a pipeline path, shrunk 10x to fit a room

Wiring (BOARD pins): L298N  IN1 29, IN2 31 (left)  IN3 35, IN4 37 (right),
GND 30.  Jumper mode (default): ENA/ENB jumpers FITTED.  PWM mode
("use_pwm": true in the config): jumpers removed, ENA 32, ENB 33, PWM enabled
with jetson-io.  IMU: SDA 3, SCL 5, VCC 1 (3.3 V), GND 6.

SAFETY: Ctrl-C, SIGTERM (sent by jetson_link_server.py) and errors stop the motors. A hard crash may NOT -- pins can
stay in their last state. Keep a physical battery switch within reach.
"""

import argparse
import csv
import json
import math
import os
import signal
import statistics
import struct
import sys
import threading
import time

CONFIG_FILE = "ugv_drive_config.json"
DEFAULTS = {
    "pins": {"left": {"en": 32, "a": 29, "b": 31}, "right": {"en": 33, "a": 35, "b": 37}},
    "use_pwm": False,          # False: ENA/ENB jumpers fitted, software speed control
    "pwm_hz": 1000,
    "invert_left": False, "invert_right": False, "swap_sides": False,
    "speed_mps": None,         # set by --calibrate-speed; required to drive
    "base_duty": 90.0,         # % while driving straight
    "turn_duty_min": 60.0,     # % -- pivots scrub the wheels; below ~60 % they stall
    "turn_duty_max": 100.0,
    "turn_kp": 2.0,            # duty % per degree of heading error when turning
    "hold_kp": 2.5,            # duty % per degree while driving straight
    "turn_tol_deg": 3.0,
    "turn_lead_s": 0.10,       # brake this far ahead of the target to allow for momentum
    "turn_nudge_deg": 10.0,    # within this, pulse power 1 tick in 3 instead of continuously
    "i2c_bus": 8, "imu_addr": 0x68,
    "bias_seconds": 30.0,      # keep the vehicle still this long at start
    "max_rate_dps": 300.0,     # gyro readings beyond this are treated as glitches
}


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def load_config():
    """Script DEFAULTS, overridden by ugv_drive_config.json if it exists.
    Prints what is actually in force: edits to DEFAULTS are silently ignored
    for any key the JSON already holds."""
    cfg = json.loads(json.dumps(DEFAULTS))
    overridden = []
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            saved = json.load(f)
        overridden = [k for k in saved if k in DEFAULTS and saved[k] != DEFAULTS[k]
                      and k not in ("speed_mps", "swap_sides", "pins")]
        cfg.update(saved)
    print("settings in force: base_duty {}%, turns {}-{}%, speed {} m/s{}".format(
        cfg["base_duty"], cfg["turn_duty_min"], cfg["turn_duty_max"], cfg["speed_mps"],
        "" if not overridden else
        "\n  NOTE: {} overrides the script for: {}".format(CONFIG_FILE, ", ".join(overridden))))
    return cfg


def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=1)


# ==========================================================================
# IMU: heading from the MPU6050 gyro
# ==========================================================================

class MPU6050Heading:
    """Integrates yaw rate in a background thread at ~500 Hz.

    - Finds which axis points up from gravity, and its sign, so the board
      can be mounted face-up or face-down.
    - Bias = MEDIAN of the yaw rate over a still period (robust to spikes).
    - Median-of-3 filter + rejection above max_rate_dps: a single corrupted
      I2C read cannot inject a large false rotation.
    """

    def __init__(self, bus=8, addr=0x68, max_rate_dps=300.0):
        try:
            from smbus import SMBus
        except ImportError:
            from smbus2 import SMBus
        self.b, self.addr, self.max_rate = SMBus(bus), addr, max_rate_dps
        self.b.write_byte_data(addr, 0x6B, 0x01)   # wake
        time.sleep(0.1)
        self.b.write_byte_data(addr, 0x1A, 0x03)   # ~44 Hz low-pass
        self.b.write_byte_data(addr, 0x1B, 0x00)   # +-250 dps
        self.b.write_byte_data(addr, 0x1C, 0x00)   # +-2 g
        time.sleep(0.05)
        self.axis, self.sign, self.bias = 2, 1.0, 0.0
        self.heading, self.rate, self.stamp = 0.0, 0.0, 0.0
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._hist = []

    def _raw(self):
        v = struct.unpack(">7h", bytes(self.b.read_i2c_block_data(self.addr, 0x3B, 14)))
        return [v[0] / 16384.0, v[1] / 16384.0, v[2] / 16384.0], [v[4] / 131.0, v[5] / 131.0, v[6] / 131.0]

    def calibrate(self, seconds):
        print("IMU: keep the vehicle completely still for {:.0f} s...".format(seconds))
        acc, gyr, t0 = [], [], time.time()
        while time.time() - t0 < seconds:
            a, g = self._raw()
            acc.append(a); gyr.append(g)
        means = [statistics.mean(a[i] for a in acc) for i in range(3)]
        self.axis = max(range(3), key=lambda i: abs(means[i]))
        self.sign = 1.0 if means[self.axis] > 0 else -1.0     # +g reads on the UP axis
        yaw = sorted(g[self.axis] for g in gyr)
        self.bias = yaw[len(yaw) // 2]
        q1, q3 = yaw[len(yaw) // 4], yaw[3 * len(yaw) // 4]
        print("IMU: up axis {}{}, yaw bias {:+.3f} deg/s, robust spread (IQR) {:.3f} deg/s, {} samples".format(
            "+" if self.sign > 0 else "-", "XYZ"[self.axis], self.bias, q3 - q1, len(yaw)))

    def _loop(self):
        last = time.time()
        while not self._stop.is_set():
            try:
                _, g = self._raw()
            except OSError:
                continue                                       # skip a failed read
            now = time.time()
            r = self.sign * (g[self.axis] - self.bias)
            self._hist = (self._hist + [r])[-3:]
            rf = sorted(self._hist)[len(self._hist) // 2]      # median of last 3
            if abs(rf) > self.max_rate:
                rf = 0.0
            with self._lock:
                self.heading += rf * (now - last)
                self.rate, self.stamp = rf, now
            last = now

    def start(self):
        self.stamp = time.time()
        threading.Thread(target=self._loop, daemon=True).start()

    def get(self):
        with self._lock:
            return self.heading, self.rate, time.time() - self.stamp

    def reset(self):
        with self._lock:
            self.heading = 0.0

    def close(self):
        self._stop.set()


# ==========================================================================
# Motors: L298N on the Jetson header
# ==========================================================================

class Motors:
    """set(left, right) takes duty in -100..100.

    PWM mode: hardware PWM on ENA/ENB.
    Jumper mode: ENA/ENB tied high by jumpers, so speed is made by switching
    the direction pins on and off in a 20 Hz cycle of 10 slots (10 % steps).
    Off-slots brake the motor (both inputs low with enable high), which is
    harsher than coasting but works without removing the jumpers.
    """

    def __init__(self, cfg):
        import Jetson.GPIO as GPIO
        self.G, self.cfg = GPIO, cfg
        self.duty = {"left": 0.0, "right": 0.0}
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        for p in cfg["pins"].values():
            GPIO.setup(p["a"], GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(p["b"], GPIO.OUT, initial=GPIO.LOW)
        self.pwm = {}
        if cfg["use_pwm"]:
            for side, p in cfg["pins"].items():
                GPIO.setup(p["en"], GPIO.OUT, initial=GPIO.LOW)
                self.pwm[side] = GPIO.PWM(p["en"], cfg["pwm_hz"])
                self.pwm[side].start(0)
        else:
            self._stop = threading.Event()
            threading.Thread(target=self._soft_pwm, daemon=True).start()

    def _dir(self, side, d):
        p = self.cfg["pins"][side]
        if self.cfg["invert_" + side]:
            d = -d
        G = self.G
        if d > 0:
            G.output(p["a"], G.HIGH); G.output(p["b"], G.LOW)
        elif d < 0:
            G.output(p["a"], G.LOW); G.output(p["b"], G.HIGH)
        else:
            G.output(p["a"], G.LOW); G.output(p["b"], G.LOW)

    def _soft_pwm(self):
        slot = 0
        while not self._stop.is_set():
            for side in ("left", "right"):
                d = self.duty[side]
                on = slot < int(abs(d) / 10.0 + 0.5)
                self._dir(side, (1 if d > 0 else -1) if on and d != 0 else 0)
            slot = (slot + 1) % 10
            time.sleep(0.005)

    def set(self, left, right):
        if self.cfg["swap_sides"]:
            left, right = right, left
        self.duty["left"], self.duty["right"] = clamp(left, -100, 100), clamp(right, -100, 100)
        if self.cfg["use_pwm"]:
            for side in ("left", "right"):
                d = self.duty[side]
                self._dir(side, d)
                self.pwm[side].ChangeDutyCycle(abs(d))

    def stop(self):
        self.duty["left"] = self.duty["right"] = 0.0
        for side in ("left", "right"):
            self._dir(side, 0)
            if side in self.pwm:
                self.pwm[side].ChangeDutyCycle(0)

    def close(self):
        self.stop()
        if not self.cfg["use_pwm"]:
            self._stop.set()
            time.sleep(0.02)
            self.stop()
        for p in self.pwm.values():
            p.stop()
        self.G.cleanup()


# ==========================================================================
# Dry-run simulator: same interfaces, no hardware
# ==========================================================================

class SimWorld:
    """A crude differential-drive robot. Its TRUE speed is deliberately 8 %
    faster than the calibrated value and the gyro has a residual bias, so the
    dry run shows realistic end-point error rather than a perfect result."""

    def __init__(self, cal_speed, base_duty):
        self.x = self.y = self.th = 0.0
        self.l = self.r = 0.0
        self.vmax = cal_speed * 1.08 / (base_duty / 100.0)   # m/s at 100 % duty
        self.wmax = 200.0                                    # deg/s at full spin
        self.t0 = time.time()
        self.sag = False                                     # field-like power loss when True
        self.w = 0.0                                         # actual yaw rate, lags the command
        self.tau = 0.15                                      # s -- rotational momentum
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        last = time.time()
        while True:
            time.sleep(0.002)
            now = time.time(); dt = now - last; last = now
            with self._lock:
                v = (self.l + self.r) / 200.0 * self.vmax
                w = (self.r - self.l) / 200.0 * self.wmax
                if self.sag and self.l * self.r < 0:          # pivoting
                    # scrubbing wheels: no rotation below a threshold that
                    # rises 1.5 %/s as the battery sags (35 % -> ~70 % at 23 s)
                    thr = 35.0 + 1.5 * (now - self.t0)
                    d = (abs(self.l) + abs(self.r)) / 2.0
                    w = 0.0 if d <= thr else w * (d - thr) / max(100.0 - thr, 1.0)
                self.w += (w - self.w) * min(dt / self.tau, 1.0)    # momentum
                w = self.w
                self.th += w * dt
                self.x += v * dt * math.cos(math.radians(self.th))
                self.y += v * dt * math.sin(math.radians(self.th))


class SimMotors:
    def __init__(self, world):
        self.w = world

    def set(self, left, right):
        with self.w._lock:
            self.w.l, self.w.r = clamp(left, -100, 100), clamp(right, -100, 100)

    def stop(self):
        self.set(0, 0)

    def close(self):
        self.stop()


class SimIMU:
    def __init__(self, world):
        self.w, self.t0, self.off = world, time.time(), 0.0

    def calibrate(self, seconds):
        print("IMU (sim): calibrated")

    def start(self):
        pass

    def get(self):
        drift = 0.05 * (time.time() - self.t0)               # 0.05 deg/s residual bias
        with self.w._lock:
            return self.w.th + drift - self.off, 0.0, 0.0

    def reset(self):
        self.off = self.get()[0] + self.off

    def close(self):
        pass


# ==========================================================================
# Controller
# ==========================================================================

class Driver:
    LOOP_HZ = 50.0

    def __init__(self, imu, motors, cfg, log_path="ugv_drive_log.csv"):
        self.imu, self.m, self.cfg = imu, motors, cfg
        self.x = self.y = 0.0
        self.turn_boost = 0.0      # extra turn power learned from stalls; kept for later turns
        self.t0 = time.time()
        self.logf = open(log_path, "w", newline="")
        self.log = csv.writer(self.logf)
        self.log.writerow(["t", "phase", "wp", "est_x", "est_y", "heading_deg",
                           "target_deg", "err_deg", "left", "right"])

    def _heading(self):
        h, _, age = self.imu.get()
        if age > 0.2:
            self.m.stop()
            raise RuntimeError("IMU data stale ({:.2f} s) -- stopped".format(age))
        return h

    def _row(self, phase, wp, h, tgt, l, r):
        self.log.writerow(["{:.3f}".format(time.time() - self.t0), phase, wp,
                           "{:.3f}".format(self.x), "{:.3f}".format(self.y),
                           "{:.2f}".format(h), "{:.2f}".format(tgt),
                           "{:.2f}".format(wrap(tgt - h)), "{:.0f}".format(l), "{:.0f}".format(r)])

    def turn_to(self, target, wp=-1, timeout=10.0):
        """Pivot to `target` degrees.

        Stopping cleanly (the field "wobble" fix):
          - inside tolerance the motors BRAKE while settling is confirmed;
            the earlier version kept driving for those ticks and overshot
          - momentum: rotating at w deg/s it coasts about w x turn_lead_s
            further after power is cut, so power is cut that early
          - within turn_nudge_deg, power is pulsed 1 tick in 3: the minimum
            power that avoids stalls is too strong for fine adjustment

        Stall handling: if the heading moves slower than 8 deg/s for 0.25 s
        while still off target, add 10 % power (battery sag; see field log).
        The boost decays by 10 % after any turn that needed none.
        """
        c = self.cfg
        stalled_this_turn = False
        for attempt in range(3):
            settled, t_end, tick = 0, time.time() + timeout, 0
            hist, last_boost = [], time.time()
            while time.time() < t_end:
                h = self._heading()
                now = time.time()
                e = wrap(target - h)
                hist = [p for p in hist + [(now, h)] if now - p[0] <= 0.3]
                recent = [p for p in hist if now - p[0] <= 0.12]
                rate = (wrap(h - recent[0][1]) / (now - recent[0][0])
                        if len(recent) > 1 and now - recent[0][0] > 0.03 else 0.0)
                toward = rate * e > 0
                coast = abs(rate) * c["turn_lead_s"] if toward else 0.0
                if abs(e) < c["turn_tol_deg"] + coast:
                    # brake; count as settled only once it has actually STOPPED
                    # inside tolerance -- while still coasting, just keep braking
                    self.m.stop()
                    if abs(e) < c["turn_tol_deg"] and abs(rate) < 5.0:
                        settled += 1
                    else:
                        settled = 0
                    self._row("settle", wp, h, target, 0, 0)
                    if settled >= 3:
                        break
                    time.sleep(1.0 / self.LOOP_HZ)
                    continue
                settled = 0
                if now - hist[0][0] >= 0.25 and now - last_boost >= 0.25:
                    prog = abs(wrap(h - hist[0][1])) / (now - hist[0][0])
                    cap = 100.0 - c["turn_duty_min"]
                    if prog < 8.0 and self.turn_boost < cap:
                        self.turn_boost = min(self.turn_boost + 10.0, cap)
                        stalled_this_turn = True
                        last_boost, hist = now, []
                        print("  turn stalled ({:.1f} deg/s, {:.0f} deg to go) -> boost {:+.0f}%".format(
                            prog, abs(e), self.turn_boost))
                base = clamp(c["turn_kp"] * abs(e), c["turn_duty_min"], c["turn_duty_max"])
                d = clamp(base + self.turn_boost, 0.0, 100.0)
                sgn = 1.0 if e > 0 else -1.0                  # e > 0: turn left (CCW)
                if abs(e) < c["turn_nudge_deg"] and tick % 3:
                    self.m.stop()                             # nudge: brake 2 ticks of 3
                    l = r = 0.0
                else:
                    l, r = -sgn * d, sgn * d
                    self.m.set(l, r)
                tick += 1
                self._row("turn", wp, h, target, l, r)
                time.sleep(1.0 / self.LOOP_HZ)
            else:
                self.m.stop()
                e = wrap(target - self._heading())
                if abs(e) <= 15.0:
                    print("  warning: turn timed out {:+.1f} deg short -- continuing; heading "
                          "hold will correct it".format(e))
                    return
                raise RuntimeError("turn stuck {:+.1f} deg from target even at {:.0f}% power -- "
                                   "battery too low or wheels slipping".format(
                                       e, clamp(c["turn_duty_max"] + self.turn_boost, 0, 100)))
            self.m.stop()
            time.sleep(0.3)                                   # let it settle, then recheck
            if abs(wrap(target - self._heading())) <= 2 * c["turn_tol_deg"]:
                break
        else:
            print("  warning: heading still {:+.1f} deg off after 3 attempts".format(
                wrap(target - self._heading())))
        if not stalled_this_turn and self.turn_boost > 0:
            self.turn_boost = max(0.0, self.turn_boost - 10.0)

    def drive(self, dist, bearing, wp=-1):
        c = self.cfg
        v = c["speed_mps"]
        t_need = dist / v
        t_start = last = time.time()
        while time.time() - t_start < t_need:
            h = self._heading()
            e = wrap(bearing - h)
            corr = clamp(c["hold_kp"] * e, -30.0, 30.0)       # e > 0: drifted right -> steer left
            l, r = c["base_duty"] - corr, c["base_duty"] + corr
            self.m.set(l, r)
            now = time.time()
            self.x += v * (now - last) * math.cos(math.radians(h))
            self.y += v * (now - last) * math.sin(math.radians(h))
            last = now
            self._row("drive", wp, h, bearing, l, r)
            # never sleep past the end: a full tick of overshoot per segment
            # adds up over many short segments
            time.sleep(max(0.0, min(1.0 / self.LOOP_HZ, t_need - (time.time() - t_start))))
        self.m.stop()
        time.sleep(0.3)

    @staticmethod
    def merge_collinear(wps, tol_deg=5.0):
        """Drop waypoints where the path barely changes direction. Each stop
        and restart costs distance (acceleration, timing), so a straight run
        of 14 waypoints should be driven as one segment, not 14."""
        pts = [(0.0, 0.0)] + list(wps)
        keep = [pts[0]]
        for i in range(1, len(pts) - 1):
            a = math.atan2(pts[i][1] - keep[-1][1], pts[i][0] - keep[-1][0])
            b = math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0])
            if abs(wrap(math.degrees(b - a))) >= tol_deg:
                keep.append(pts[i])
        keep.append(pts[-1])
        return keep[1:]

    def follow(self, wps):
        n0 = len(wps)
        wps = self.merge_collinear(wps)
        if len(wps) < n0:
            print("merged {} collinear waypoints -> {} segments".format(n0, len(wps)))
        print("\n wp   target (x, y)     dist   bearing   est. after (x, y)   heading")
        for i, (tx, ty) in enumerate(wps):
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)
            if dist < 0.05:
                continue
            bearing = math.degrees(math.atan2(dy, dx))
            self.turn_to(bearing, i)
            self.drive(dist, bearing, i)
            print(" {:>2}   ({:>5.2f}, {:>5.2f})   {:>5.2f}   {:>+7.1f}   ({:>5.2f}, {:>5.2f})      {:>+7.1f}".format(
                i, tx, ty, dist, bearing, self.x, self.y, self._heading()))

    def close(self):
        self.logf.close()


# ==========================================================================
# Waypoints
# ==========================================================================

DEMOS = {
    "line": [(1.0, 0.0)],
    "square": [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
    "L": [(1.0, 0.0), (1.0, 1.0)],
}


def load_waypoints(path, align):
    """path_msg.json from the pipeline, or a CSV of x,y lines. Pipeline paths
    are in the drone's world frame: shift so the first waypoint is (0, 0)
    and, with align='first-segment', rotate so the first leg points along
    the vehicle's starting direction (+x)."""
    if path.endswith(".json"):
        with open(path) as f:
            pts = [(w[0], w[1]) for w in json.load(f)["waypoints"]]
    else:
        pts = []
        with open(path) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    x, y = line.replace(",", " ").split()[:2]
                    pts.append((float(x), float(y)))
    x0, y0 = pts[0]
    pts = [(x - x0, y - y0) for x, y in pts]
    if align == "first-segment" and len(pts) > 1:
        a = -math.atan2(pts[1][1], pts[1][0])
        ca, sa = math.cos(a), math.sin(a)
        pts = [(x * ca - y * sa, x * sa + y * ca) for x, y in pts]
    return pts[1:] if pts and pts[0] == (0.0, 0.0) else pts


# ==========================================================================

def build(cfg, dry):
    if dry:
        if cfg["speed_mps"] is None:
            cfg["speed_mps"] = 0.4
        w = SimWorld(cfg["speed_mps"], cfg["base_duty"])
        return SimIMU(w), SimMotors(w), w
    imu = MPU6050Heading(cfg["i2c_bus"], cfg["imu_addr"], cfg["max_rate_dps"])
    return imu, Motors(cfg), None


def _stop_signal(signum, frame):
    raise KeyboardInterrupt      # same path as Ctrl-C: the finally block turns the motors off


def main():
    # Stop requests from the Jetson link server arrive as SIGTERM. SIGINT is
    # reset too: a program started in the background can inherit "ignore Ctrl-C".
    signal.signal(signal.SIGTERM, _stop_signal)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dry-run", action="store_true", help="simulate, no hardware")
    ap.add_argument("--check-turn", action="store_true")
    ap.add_argument("--calibrate-speed", type=float, metavar="SECONDS")
    ap.add_argument("--demo", choices=sorted(DEMOS))
    ap.add_argument("--waypoints", help="path_msg.json or x,y CSV")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all waypoints")
    ap.add_argument("--align", choices=("first-segment", "none"), default="first-segment")
    ap.add_argument("--face-start", action="store_true",
                    help="after the last waypoint, turn back to the starting direction")
    a = ap.parse_args()

    cfg = load_config()
    if not (a.dry_run or a.check_turn or a.calibrate_speed) and cfg["speed_mps"] is None:
        sys.exit("No speed calibration yet. Run --calibrate-speed 3 first.")

    imu, motors, world = build(cfg, a.dry_run)
    drv = None
    try:
        imu.calibrate(2.0 if a.dry_run else cfg["bias_seconds"])
        imu.start()
        time.sleep(0.2)
        imu.reset()
        drv = Driver(imu, motors, cfg)

        if a.check_turn:
            print("spinning LEFT (counter-clockwise) for 0.8 s...")
            h0 = imu.get()[0]
            motors.set(-cfg["turn_duty_max"], cfg["turn_duty_max"])
            time.sleep(0.8)
            motors.stop(); time.sleep(0.3)
            d = wrap(imu.get()[0] - h0)
            print("heading changed {:+.1f} deg".format(d))
            if d > 10:
                print("PASS: motors and IMU agree. Ready to calibrate speed.")
            elif d < -10:
                print("FAIL: it turned the wrong way for the IMU. Set \"swap_sides\": true in {}"
                      " (or swap the two motors' wires), then rerun.".format(CONFIG_FILE))
                cfg["swap_sides"] = not cfg["swap_sides"]; save_config(cfg)
                print("      (done for you: swap_sides is now {})".format(cfg["swap_sides"]))
            else:
                print("FAIL: barely rotated. Raise turn_duty_max, check battery, or check "
                      "that the wheels are on the ground.")
            return

        if a.calibrate_speed:
            print("driving straight for {:.1f} s at {:.0f}% -- mark the start point".format(
                a.calibrate_speed, cfg["base_duty"]))
            cfg_tmp = dict(cfg, speed_mps=1.0)                # time = distance with v = 1
            drv.cfg = cfg_tmp
            drv.drive(a.calibrate_speed, 0.0)
            m = input("Measured distance travelled in metres: ").strip()
            cfg["speed_mps"] = float(m) / a.calibrate_speed
            save_config(cfg)
            print("saved speed_mps = {:.3f} m/s to {}".format(cfg["speed_mps"], CONFIG_FILE))
            print("Recalibrate if you change base_duty, the battery, or the floor surface.")
            return

        wps = DEMOS[a.demo] if a.demo else load_waypoints(a.waypoints, a.align)
        wps = [(x * a.scale, y * a.scale) for x, y in wps]
        print("following {} waypoints, speed {:.3f} m/s".format(len(wps), cfg["speed_mps"]))
        drv.follow(wps)
        if a.face_start:
            drv.turn_to(0.0, len(wps))
            print("turned back to the starting direction: heading {:+.1f} deg".format(drv._heading()))
        ex, ey = wps[-1]
        print("\nfinal waypoint      ({:.2f}, {:.2f})".format(ex, ey))
        print("estimated position  ({:.2f}, {:.2f})   (dead reckoning)".format(drv.x, drv.y))
        if world is not None:
            print("TRUE position (sim) ({:.2f}, {:.2f})   error {:.2f} m".format(
                world.x, world.y, math.hypot(world.x - ex, world.y - ey)))
        else:
            print("Now tape-measure where it actually stopped, relative to the start mark.\n"
                  "That gap is the real accuracy; the estimate above does not know about it.")
        print("log: ugv_drive_log.csv")
    except KeyboardInterrupt:
        print("\nstopped by Ctrl-C")
    finally:
        motors.stop()
        motors.close()
        imu.close()
        if drv:
            drv.close()
        print("motors stopped")


if __name__ == "__main__":
    main()
