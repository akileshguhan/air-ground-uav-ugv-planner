#!/usr/bin/env python3
"""
arm_disarm_test.py -- arm the PX4 drone from the Raspberry Pi through MAVROS,
keep it armed for a few seconds, then disarm. A bench test of the
Pi -> MAVROS -> TELEM2 -> Pixhawk command path.

    PROPELLERS OFF. Arming spins the motors (PX4 idles them when armed).

What it does
  1. waits for MAVROS to connect to the FCU, prints mode / armed state
  2. optionally switches flight mode (--mode), e.g. MANUAL or STABILIZED
  3. sends ARM, waits until PX4 reports armed
  4. holds for --hold seconds, then sends DISARM and waits for it
  PX4's own messages (e.g. "Arming denied: ...") are printed as they arrive,
  so a refusal tells you why. Ctrl-C at any point sends DISARM.

Before running: MAVROS must be running in another terminal:
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyAMA0:921600

USAGE
  source /opt/ros/jazzy/setup.bash
  python3 arm_disarm_test.py                  # arm, 5 s, disarm
  python3 arm_disarm_test.py --hold 3 --mode MANUAL
  python3 arm_disarm_test.py --status         # just print state, no arming

Notes
  * PX4 auto-disarms after COM_DISARM_PRFLT seconds (default 10) if it does
    not take off, so keep --hold below that.
  * Arming in POSCTL / OFFBOARD needs a position estimate (GPS or flow);
    MANUAL or STABILIZED does not. Use --mode MANUAL for a bench test.
  * If an RC transmitter is on, its mode switch can override --mode.
"""
import argparse
import sys
import threading
import time


def main(argv=None):
    p = argparse.ArgumentParser(description="Arm, hold, disarm a PX4 drone via MAVROS.")
    p.add_argument("--hold", type=float, default=5.0, help="seconds to stay armed (keep < 10)")
    p.add_argument("--mode", help="switch to this mode first, e.g. MANUAL, STABILIZED, POSCTL")
    p.add_argument("--status", action="store_true", help="only print the current state")
    p.add_argument("--timeout", type=float, default=10.0, help="s to wait for each step")
    a = p.parse_args(argv)
    if a.hold >= 10:
        print("note: PX4 normally auto-disarms after 10 s on the ground; it may disarm before --hold ends")

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from mavros_msgs.msg import State, StatusText
    from mavros_msgs.srv import CommandBool, SetMode

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)   # Ctrl-C -> KeyboardInterrupt
    node = rclpy.create_node("arm_disarm_test")
    st = {"state": State(), "got": False}

    def on_state(msg):
        if st["got"] and (msg.armed != st["state"].armed or msg.mode != st["state"].mode):
            print("  [state] mode %s, %s" % (msg.mode, "ARMED" if msg.armed else "disarmed"))
        st["state"], st["got"] = msg, True

    def on_text(msg):
        print("  [PX4] %s" % msg.text)

    node.create_subscription(State, "/mavros/state", on_state, qos_profile_sensor_data)
    node.create_subscription(StatusText, "/mavros/statustext/recv", on_text, qos_profile_sensor_data)
    arm_cli = node.create_client(CommandBool, "/mavros/cmd/arming")
    mode_cli = node.create_client(SetMode, "/mavros/set_mode")
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    def call(cli, req):
        fut = cli.call_async(req)
        end = time.time() + 3.0
        while not fut.done() and time.time() < end:
            time.sleep(0.02)
        return fut.result() if fut.done() else None

    def wait_for(cond, what):
        end = time.time() + a.timeout
        while time.time() < end:
            if cond():
                return True
            time.sleep(0.05)
        print("  timed out waiting for %s" % what)
        return False

    def arm(value):
        req = CommandBool.Request()
        req.value = value
        r = call(arm_cli, req)
        if r is None:
            print("  no reply from /mavros/cmd/arming")
            return False
        if not r.success:
            print("  %s REFUSED by PX4 (result %d) -- see [PX4] messages above"
                  % ("ARM" if value else "DISARM", r.result))
            return False
        return wait_for(lambda: st["state"].armed == value, "PX4 to report %s" % ("armed" if value else "disarmed"))

    armed_by_us = False
    rc = 1
    try:
        print("waiting for MAVROS services and FCU connection ...")
        for cli, name in ((arm_cli, "/mavros/cmd/arming"), (mode_cli, "/mavros/set_mode")):
            if not cli.wait_for_service(timeout_sec=a.timeout):
                print("service %s not available -- is MAVROS running?" % name)
                return 1
        if not wait_for(lambda: st["got"] and st["state"].connected, "FCU connection (check TELEM2 wiring/baud)"):
            return 1
        s = st["state"]
        print("connected: mode %s, %s, system status %d" % (s.mode, "ARMED" if s.armed else "disarmed", s.system_status))
        if a.status:
            return 0
        if s.armed:
            print("already armed -- not touching it. Disarm from the RC/QGC first.")
            return 1

        if a.mode and s.mode != a.mode:
            req = SetMode.Request()
            req.custom_mode = a.mode
            r = call(mode_cli, req)
            if r is None or not r.mode_sent or not wait_for(lambda: st["state"].mode == a.mode, "mode " + a.mode):
                print("could not switch to %s (an RC mode switch can override this)" % a.mode)
                return 1
            print("mode is now %s" % a.mode)

        print("ARMING (propellers must be OFF) ...")
        if not arm(True):
            return 1
        armed_by_us = True
        print("ARMED. Holding %.1f s ..." % a.hold)
        end = time.time() + a.hold
        while time.time() < end:
            if not st["state"].armed:
                print("PX4 disarmed by itself (auto-disarm or RC)")
                armed_by_us = False
                return 0
            time.sleep(0.1)

        print("DISARMING ...")
        if arm(False):
            armed_by_us = False
            print("DISARMED. Test passed.")
            rc = 0
        return rc
    except KeyboardInterrupt:
        print("\nCtrl-C")
        return 1
    finally:
        if armed_by_us and st["state"].armed:
            print("sending DISARM before exit ...")
            req = CommandBool.Request()
            req.value = False
            call(arm_cli, req)
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
