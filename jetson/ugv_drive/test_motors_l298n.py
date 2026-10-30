#!/usr/bin/env python3
"""
test_motors_l298n.py -- test two DC motors on an L298N driven straight from
the Jetson Xavier NX 40-pin header.

WHEELS OFF THE GROUND FOR EVERY TEST. Keep a hand on the battery switch.

Wiring (BOARD pin numbers):
    ENA -> 32 (PWM)   IN1 -> 29   IN2 -> 31      left motor
    ENB -> 33 (PWM)   IN3 -> 35   IN4 -> 37      right motor
    GND -> 30 and battery negative (common ground)
    ENA/ENB jumpers REMOVED on the L298N. 5V terminal left unconnected.

Before running: enable PWM on pins 32 and 33 with
    sudo /opt/nvidia/jetson-io/jetson-io.py      (then reboot)

Stages:
    0  hardware check: board model, each PWM pin starts cleanly
    1  each motor, each direction, one at a time -- you confirm what you see
    2  deadband: the duty cycle at which each motor actually starts turning
    3  combined moves: forward, reverse, spin left, spin right

Results (direction inversions, deadband) are saved to motor_test_result.json
for the motor controller to use later.

SAFETY: Ctrl-C and ordinary errors stop both motors (finally block + signal
handler). A HARD crash or kill -9 does NOT -- the Jetson's PWM is hardware
generated and keeps running at its last value. That is why the wheels must
be off the ground and a physical battery switch must be within reach.
"""

import json
import select
import signal
import sys
import time

try:
    import Jetson.GPIO as GPIO
except ImportError:
    sys.exit("Jetson.GPIO not found. On JetPack: sudo apt install python3-jetson-gpio")

# ---- configuration --------------------------------------------------------
PINS = {
    "left":  {"en": 32, "a": 29, "b": 31},
    "right": {"en": 33, "a": 35, "b": 37},
}
PWM_HZ = 1000          # L298 switches slowly; 1-5 kHz is its comfortable range
TEST_DUTY = 50.0       # % for direction tests
RAMP_MS = 400          # soft start/stop, avoids current spikes on the battery
RESULT_FILE = "motor_test_result.json"

pwm = {}
invert = {"left": False, "right": False}


# ---- low-level ------------------------------------------------------------
def all_stop():
    """Coast both motors: duty 0 and both direction pins low."""
    for side, p in PINS.items():
        if side in pwm:
            try:
                pwm[side].ChangeDutyCycle(0)
            except Exception:
                pass
        for k in ("a", "b"):
            try:
                GPIO.output(p[k], GPIO.LOW)
            except Exception:
                pass


def set_motor(side, duty):
    """duty in -100..100. Sign = direction, magnitude = PWM %."""
    p = PINS[side]
    d = -duty if invert[side] else duty
    if d > 0:
        GPIO.output(p["a"], GPIO.HIGH); GPIO.output(p["b"], GPIO.LOW)
    elif d < 0:
        GPIO.output(p["a"], GPIO.LOW); GPIO.output(p["b"], GPIO.HIGH)
    else:
        GPIO.output(p["a"], GPIO.LOW); GPIO.output(p["b"], GPIO.LOW)
    pwm[side].ChangeDutyCycle(min(abs(duty), 100.0))


def ramp(targets, ms=RAMP_MS):
    """Linearly ramp {side: duty} from 0 to target."""
    steps = 10
    for i in range(1, steps + 1):
        for side, t in targets.items():
            set_motor(side, t * i / steps)
        time.sleep(ms / 1000.0 / steps)


def run(targets, seconds):
    ramp(targets)
    time.sleep(seconds)
    ramp({s: 0 for s in targets}, ms=RAMP_MS // 2)
    all_stop()


def ask(prompt):
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return ""


def shutdown(*_):
    print("\nstopping motors...")
    all_stop()
    for p in pwm.values():
        try:
            p.stop()
        except Exception:
            pass
    GPIO.cleanup()
    sys.exit(0)


# ---- stages ---------------------------------------------------------------
def stage0():
    print("\n=== 0. hardware check ===")
    print("board model: {}   Jetson.GPIO {}".format(GPIO.model, GPIO.VERSION))
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    ok = True
    for side, p in PINS.items():
        GPIO.setup(p["a"], GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(p["b"], GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(p["en"], GPIO.OUT, initial=GPIO.LOW)
        try:
            pwm[side] = GPIO.PWM(p["en"], PWM_HZ)
            pwm[side].start(0)
            print("  [PASS] {} motor PWM on pin {}".format(side, p["en"]))
        except Exception as e:
            ok = False
            print("  [FAIL] {} motor PWM on pin {}: {}".format(side, p["en"], e))
    if not ok:
        print("\nA PWM pin failed. Run sudo /opt/nvidia/jetson-io/jetson-io.py,\n"
              "enable pwm on pins 32 and 33, save and reboot.\n"
              "If pin 33 still fails (a known issue on some Xavier NX boards),\n"
              "refit the ENB jumper and drive that motor at full speed only.")
        shutdown()


def stage1():
    print("\n=== 1. direction check (one motor at a time, {}% for 2 s) ===".format(TEST_DUTY))
    print("For each run, say which way the wheel turned.")
    for side in ("left", "right"):
        if ask("\nReady to spin the {} motor FORWARD? [Enter = go, s = skip] ".format(side)) == "s":
            continue
        run({side: TEST_DUTY}, 2.0)
        r = ask("Did the {} wheel turn the robot's FORWARD way? [y/n/none] ".format(side))
        if r.startswith("n"):
            invert[side] = True
            print("  -> will invert {} motor (or swap its two wires on OUT)".format(side))
        elif r.startswith("none") or r == "":
            print("  -> no movement: check ENA/ENB jumper removed, motor wires, battery,\n"
                  "     and common ground between battery negative and Jetson pin 30.")
        run({side: -TEST_DUTY}, 2.0)
        ask("Did it now turn BACKWARD? (just confirm) [Enter] ")


def stage2():
    print("\n=== 2. deadband: duty where each motor starts turning ===")
    print("The duty rises slowly. Press Enter the moment the wheel starts turning.")
    deadband = {}
    for side in ("left", "right"):
        if ask("\nReady for {} motor? [Enter = go, s = skip] ".format(side)) == "s":
            continue
        found = None
        GPIO.output(PINS[side]["a"], GPIO.LOW)
        for duty in range(0, 101, 2):
            set_motor(side, duty)
            print("\r  duty {:>3}%".format(duty), end="", flush=True)
            r, _, _ = select.select([sys.stdin], [], [], 0.35)
            if r:
                sys.stdin.readline()
                found = duty
                break
        all_stop()
        print()
        deadband[side] = found
        print("  {} motor starts at {}%".format(side, found if found is not None else ">100 (did not start)"))
    return deadband


def stage3():
    print("\n=== 3. combined moves ({}%, 1.5 s each) ===".format(TEST_DUTY))
    moves = [("forward", TEST_DUTY, TEST_DUTY), ("reverse", -TEST_DUTY, -TEST_DUTY),
             ("spin left", -TEST_DUTY, TEST_DUTY), ("spin right", TEST_DUTY, -TEST_DUTY)]
    for name, l, r in moves:
        if ask("\n{}: [Enter = go, s = skip] ".format(name)) == "s":
            continue
        run({"left": l, "right": r}, 1.5)
        ask("  looked right? [Enter] ")


def main():
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(__doc__.split("Stages:")[0])
    if ask("Wheels off the ground and battery switch in reach? [yes/no] ") != "yes":
        print("Not starting."); return
    try:
        stage0()
        stage1()
        deadband = stage2()
        stage3()
        with open(RESULT_FILE, "w") as f:
            json.dump({"pins": PINS, "pwm_hz": PWM_HZ, "invert": invert,
                       "deadband_percent": deadband}, f, indent=1)
        print("\nsaved {}".format(RESULT_FILE))
        print("invert: {}   deadband: {}".format(invert, deadband))
    finally:
        all_stop()
        for p in pwm.values():
            try:
                p.stop()
            except Exception:
                pass
        GPIO.cleanup()
        print("motors stopped, GPIO released")


if __name__ == "__main__":
    main()
