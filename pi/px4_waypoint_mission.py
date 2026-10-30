#!/usr/bin/env python3
"""
px4_waypoint_mission.py -- fly a PX4 drone through a list of waypoints in
OFFBOARD mode, from a Raspberry Pi running ROS 2 (Jazzy) + MAVROS.

    Pi (this script) --MAVROS--> Pixhawk TELEM2 --> PX4

What it does
  1. waits for MAVROS to connect to the FCU and for a local position estimate
  2. records the takeoff point and the nose direction
  3. streams setpoints, switches to OFFBOARD, arms
  4. climbs to --altitude, flies each waypoint at --speed holding the takeoff yaw
  5. returns above the takeoff point (unless --no-return) and lands (AUTO.LAND)

WAYPOINTS
  path_msg.json-style ({"waypoints": [[x,y,z], ...]}) or a CSV of "x,y" rows,
  in metres. They are made relative to the first waypoint, so the drone treats
  its takeoff spot as the path's first point.
  Frame "body-start" (default): +x = where the nose pointed at takeoff, +y = left.
  Frame "enu": +x = East, +y = North (MAVROS local frame axes).
  The file's z is IGNORED (the planner's z is ground height); every waypoint is
  flown at --altitude above the takeoff point.

SPEED
  If PX4 is handed a far-away position setpoint it flies there at MPC_XY_CRUISE
  (5 m/s by default). So the setpoint is slid along each leg ("carrot") at
  --speed, and the drone follows it at that speed.

SAFETY (read before flying)
  * Test in PX4 SITL first, then props OFF on the bench, then a low first flight.
  * Keep an RC transmitter on, with a mode switch and a kill switch. If the
    pilot switches out of OFFBOARD, this script stops sending commands and
    exits (hands-off) -- the pilot has control.
  * Set PX4 offboard-loss failsafe: COM_OF_LOSS_T (e.g. 1 s), COM_OBL_RC_ACT.
  * Defaults are deliberately slow and low. The plan is refused if it leaves
    --max-radius or exceeds --max-alt.

USAGE
  python3 px4_waypoint_mission.py --waypoints path_lane_change.json --scale 0.2 --plan-only
  python3 px4_waypoint_mission.py --demo square --scale 3
  python3 px4_waypoint_mission.py --waypoints path_msg.json --altitude 2.5 --speed 0.5

Run after:  source /opt/ros/jazzy/setup.bash
and with MAVROS already running in another terminal:
  ros2 launch mavros px4.launch fcu_url:=/dev/ttyAMA0:921600
Python 3.8+ compatible.
"""
import argparse
import csv
import json
import math
import sys
import time

RATE_HZ = 20.0

DEMOS = {
    "square": [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
    "line": [(1.0, 0.0), (0.0, 0.0)],
}


class Abort(Exception):
    """Something is unsafe: land (or hold) and stop."""


class PilotTakeover(Exception):
    """Pilot switched out of OFFBOARD: stop commanding, hands off."""


# ---------------------------------------------------------------------------
# Pure helpers (no ROS) -- testable on any machine
# ---------------------------------------------------------------------------

def load_waypoints(path):
    """Read waypoints -> list of (x, y) relative to the first point."""
    pts = []
    if path.lower().endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        raw = data["waypoints"] if isinstance(data, dict) else data
        for w in raw:
            pts.append((float(w[0]), float(w[1])))
    else:
        with open(path) as f:
            for row in csv.reader(f):
                if not row or row[0].strip().startswith("#"):
                    continue
                try:
                    pts.append((float(row[0]), float(row[1])))
                except ValueError:
                    continue  # header line
    if not pts:
        raise ValueError("no waypoints found in %s" % path)
    x0, y0 = pts[0]
    rel = [(x - x0, y - y0) for x, y in pts]
    # The first point is the takeoff spot itself; don't fly to it.
    while rel and math.hypot(rel[0][0], rel[0][1]) < 1e-6:
        rel.pop(0)
    if not rel:
        raise ValueError("all waypoints are at the start point")
    return rel


def merge_collinear(rel, angle_tol_deg=5.0):
    """Drop intermediate points that lie on a straight leg, so the drone
    doesn't stop and hold at every 2 m sample of a straight line."""
    pts = [(0.0, 0.0)] + list(rel)
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = out[-1], pts[i], pts[i + 1]
        h1 = math.atan2(b[1] - a[1], b[0] - a[0])
        h2 = math.atan2(c[1] - b[1], c[0] - b[0])
        d = abs((h2 - h1 + math.pi) % (2 * math.pi) - math.pi)
        if math.degrees(d) > angle_tol_deg:
            out.append(b)
    out.append(pts[-1])
    return out[1:]


def to_local(rel_xy, start_xyz, yaw0, altitude, frame, scale):
    """Relative (x, y) -> MAVROS local ENU (x, y, z) targets."""
    out = []
    c, s = math.cos(yaw0), math.sin(yaw0)
    for x, y in rel_xy:
        x, y = x * scale, y * scale
        if frame == "body-start":
            x, y = c * x - s * y, s * x + c * y
        out.append((start_xyz[0] + x, start_xyz[1] + y, start_xyz[2] + altitude))
    return out


def check_plan(rel_xy, scale, altitude, max_radius, max_alt, min_alt=1.0):
    problems = []
    if altitude < min_alt:
        problems.append("altitude %.1f m is below the %.1f m minimum" % (altitude, min_alt))
    if altitude > max_alt:
        problems.append("altitude %.1f m exceeds --max-alt %.1f m" % (altitude, max_alt))
    far = max(math.hypot(x, y) * scale for x, y in rel_xy)
    if far > max_radius:
        problems.append("farthest waypoint is %.1f m from takeoff, beyond --max-radius %.1f m"
                        % (far, max_radius))
    return problems, far


def carrot(a, b, travelled):
    """Point `travelled` metres from a toward b (clamped at b)."""
    d = [b[i] - a[i] for i in range(3)]
    length = math.sqrt(sum(v * v for v in d))
    if length < 1e-6 or travelled >= length:
        return tuple(b), True
    f = travelled / length
    return tuple(a[i] + f * d[i] for i in range(3)), False


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# ---------------------------------------------------------------------------
# ROS 2 / MAVROS mission
# ---------------------------------------------------------------------------

class Rate(object):
    """Fixed-period loop timer (the ROS executor spins in its own thread)."""
    def __init__(self, hz):
        self.period = 1.0 / hz
        self.next_t = time.monotonic()

    def sleep(self):
        self.next_t += self.period
        delay = self.next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_t = time.monotonic()  # fell behind; don't try to catch up


class Mission(object):
    def __init__(self, args):
        import threading
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from geometry_msgs.msg import PoseStamped
        from mavros_msgs.msg import State
        from mavros_msgs.srv import CommandBool, SetMode
        from sensor_msgs.msg import BatteryState

        self.rclpy = rclpy
        self.PoseStamped = PoseStamped
        self.CommandBool = CommandBool
        self.SetMode = SetMode
        self.args = args

        from rclpy.signals import SignalHandlerOptions
        # Keep Ctrl-C as a Python KeyboardInterrupt: ROS stays up so the
        # emergency handler can still command AUTO.LAND.
        if not rclpy.ok():
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.node = rclpy.create_node("px4_waypoint_mission")
        self.state = State()
        self.pose = None
        self.pose_t = 0.0
        self.battery = None
        self.in_offboard = False
        self.q0 = None

        # Best-effort subscriptions connect to both best-effort and reliable
        # MAVROS publishers, so they work whatever QoS each plugin uses.
        n = self.node
        n.create_subscription(State, "/mavros/state", self._on_state, qos_profile_sensor_data)
        n.create_subscription(PoseStamped, "/mavros/local_position/pose", self._on_pose,
                              qos_profile_sensor_data)
        n.create_subscription(BatteryState, "/mavros/battery", self._on_battery,
                              qos_profile_sensor_data)
        self.sp_pub = n.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 10)
        self.arm_cli = n.create_client(CommandBool, "/mavros/cmd/arming")
        self.mode_cli = n.create_client(SetMode, "/mavros/set_mode")

        # Callbacks run in this background thread; the mission runs in main.
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(n)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        self.info("waiting for MAVROS services ...")
        for cli, name in ((self.arm_cli, "/mavros/cmd/arming"), (self.mode_cli, "/mavros/set_mode")):
            if not cli.wait_for_service(timeout_sec=30.0):
                raise Abort("service %s not available -- is MAVROS running?" % name)
        self.rate = Rate(RATE_HZ)

        self.log_f = open(args.log, "w")
        self.log = csv.writer(self.log_f)
        self.log.writerow(["t", "phase", "wp", "sp_x", "sp_y", "sp_z",
                           "x", "y", "z", "dist_m", "mode", "armed"])
        self.t0 = time.time()

    # -- logging --
    def info(self, msg):
        self.node.get_logger().info(msg)

    def warn(self, msg):
        self.node.get_logger().warn(msg)

    def error(self, msg):
        self.node.get_logger().error(msg)

    # -- callbacks --
    def _on_state(self, msg):
        self.state = msg

    def _on_pose(self, msg):
        self.pose = msg
        self.pose_t = time.time()

    def _on_battery(self, msg):
        self.battery = msg

    # -- services (non-blocking request, short wait for the reply) --
    def _call(self, cli, req, timeout=2.0):
        fut = cli.call_async(req)
        end = time.time() + timeout
        while not fut.done() and time.time() < end:
            time.sleep(0.01)
        return fut.result() if fut.done() else None

    def set_mode(self, mode):
        req = self.SetMode.Request()
        req.custom_mode = mode
        return self._call(self.mode_cli, req)

    def arm(self, value=True):
        req = self.CommandBool.Request()
        req.value = value
        return self._call(self.arm_cli, req)

    # -- basics --
    def pos(self):
        p = self.pose.pose.position
        return (p.x, p.y, p.z)

    def send(self, sp):
        m = self.PoseStamped()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.pose.position.x = float(sp[0])
        m.pose.position.y = float(sp[1])
        m.pose.position.z = float(sp[2])
        m.pose.orientation = self.q0
        self.sp_pub.publish(m)

    def check(self):
        if not self.rclpy.ok():
            raise Abort("ROS shutdown")
        if self.pose is None or time.time() - self.pose_t > 0.5:
            raise Abort("local position lost (no pose for > 0.5 s)")
        if self.in_offboard:
            if self.state.mode != "OFFBOARD":
                raise PilotTakeover("mode changed to %s" % self.state.mode)
            if not self.state.armed:
                raise Abort("vehicle disarmed in flight")

    def tick(self, sp, phase, wp):
        self.check()
        self.send(sp)
        p = self.pos()
        d = math.sqrt(sum((sp[i] - p[i]) ** 2 for i in range(3)))
        self.log.writerow(["%.2f" % (time.time() - self.t0), phase, wp,
                           "%.3f" % sp[0], "%.3f" % sp[1], "%.3f" % sp[2],
                           "%.3f" % p[0], "%.3f" % p[1], "%.3f" % p[2],
                           "%.3f" % d, self.state.mode, int(self.state.armed)])
        self.rate.sleep()
        return p

    def wait_until(self, cond, timeout, what):
        end = time.time() + timeout
        while not cond():
            if not self.rclpy.ok():
                raise Abort("ROS shutdown")
            if time.time() > end:
                raise Abort("timed out waiting for " + what)
            self.rate.sleep()

    # -- phases --
    def preflight(self):
        self.info("waiting for FCU connection ...")
        self.wait_until(lambda: self.state.connected, 30, "FCU connection")
        self.info("waiting for local position estimate ...")
        self.wait_until(lambda: self.pose is not None and time.time() - self.pose_t < 0.5,
                        30, "local position (/mavros/local_position/pose)")
        if self.battery is not None and self.battery.percentage >= 0:
            pct = self.battery.percentage * 100.0
            self.info("battery %.0f%%" % pct)
            if pct < self.args.min_battery:
                raise Abort("battery %.0f%% below --min-battery %.0f%%" % (pct, self.args.min_battery))
        else:
            self.warn("no battery reading -- check the battery yourself")
        if self.state.armed:
            raise Abort("vehicle is already armed; start this script on the ground, disarmed")

    def engage(self, hold_sp):
        self.info("streaming setpoints before OFFBOARD ...")
        for _ in range(int(2 * RATE_HZ)):
            self.check()
            self.send(hold_sp)
            self.rate.sleep()
        end = time.time() + 15.0
        last = 0.0
        while not (self.state.mode == "OFFBOARD" and self.state.armed):
            self.check()
            if time.time() > end:
                raise Abort("could not enter OFFBOARD and arm within 15 s "
                            "(check pre-arm messages in QGroundControl)")
            if time.time() - last > 1.0:
                last = time.time()
                if self.state.mode != "OFFBOARD":
                    self.info("requesting OFFBOARD")
                    self.set_mode("OFFBOARD")
                elif not self.state.armed:
                    self.info("arming")
                    self.arm(True)
            self.send(hold_sp)
            self.rate.sleep()
        self.in_offboard = True
        self.info("OFFBOARD + armed")

    def fly_to(self, a, b, speed, phase, wp):
        a_ = self.args
        step = speed / RATE_HZ
        travelled = 0.0
        reached = False
        while not reached:
            sp, reached = carrot(a, b, travelled)
            self.tick(sp, phase, wp)
            travelled += step
        inside_since = None
        end = time.time() + a_.wp_timeout
        while True:
            p = self.tick(b, phase, wp)
            ok = (math.hypot(b[0] - p[0], b[1] - p[1]) <= a_.accept_xy
                  and abs(b[2] - p[2]) <= a_.accept_z)
            now = time.time()
            if ok:
                inside_since = inside_since or now
                if now - inside_since >= a_.hold:
                    return
            else:
                inside_since = None
            if now > end:
                raise Abort("waypoint %s not reached within %.0f s (at %.2f,%.2f,%.2f)"
                            % (wp, a_.wp_timeout, p[0], p[1], p[2]))

    def land(self):
        self.in_offboard = False
        self.info("landing (AUTO.LAND)")
        self.set_mode("AUTO.LAND")
        self.wait_until(lambda: not self.state.armed, 120, "landing / disarm")
        self.info("landed and disarmed")

    def run(self, rel):
        a_ = self.args
        self.preflight()
        start = self.pos()
        self.q0 = self.pose.pose.orientation
        yaw0 = yaw_from_quat(self.q0)
        self.info("takeoff point (%.2f, %.2f, %.2f), heading %.1f deg (ENU)"
                  % (start[0], start[1], start[2], math.degrees(yaw0)))
        targets = to_local(rel, start, yaw0, a_.altitude, a_.frame, a_.scale)
        top = (start[0], start[1], start[2] + a_.altitude)

        self.engage(start)
        self.info("climbing to %.1f m" % a_.altitude)
        self.fly_to(start, top, a_.climb_speed, "climb", 0)
        prev = top
        for i, t in enumerate(targets, 1):
            self.info("waypoint %d/%d -> (%.2f, %.2f, %.2f)" % (i, len(targets), t[0], t[1], t[2]))
            self.fly_to(prev, t, a_.speed, "leg", i)
            prev = t
        if not a_.no_return:
            self.info("returning above takeoff point")
            self.fly_to(prev, top, a_.speed, "return", 0)
        self.land()

    def emergency(self, reason):
        self.error("ABORT: %s" % reason)
        self.in_offboard = False
        try:
            if self.state.armed and self.state.mode == "OFFBOARD":
                mode = "AUTO.LOITER" if self.args.on_abort == "hold" else "AUTO.LAND"
                self.error("switching to %s" % mode)
                if self.set_mode(mode) is None:
                    self.error("no reply to mode change -- TAKE OVER ON RC")
        except Exception as e:  # context may already be shut down
            self.error("could not change mode: %s -- TAKE OVER ON RC" % e)

    def close(self):
        try:
            self.log_f.close()
        except Exception:
            pass
        try:
            self.executor.shutdown()
            self.node.destroy_node()
            if self.rclpy.ok():
                self.rclpy.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Fly PX4 through waypoints in OFFBOARD via MAVROS.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--waypoints", help="path_msg.json-style file or CSV x,y")
    src.add_argument("--demo", choices=sorted(DEMOS), help="built-in shape (unit size, use --scale)")
    p.add_argument("--scale", type=float, default=1.0, help="multiply waypoint x,y by this")
    p.add_argument("--altitude", type=float, default=2.0, help="m above takeoff point")
    p.add_argument("--speed", type=float, default=0.5, help="horizontal speed m/s")
    p.add_argument("--climb-speed", type=float, default=0.5, help="m/s")
    p.add_argument("--frame", choices=["body-start", "enu"], default="body-start")
    p.add_argument("--accept-xy", type=float, default=0.3, help="m, waypoint reached radius")
    p.add_argument("--accept-z", type=float, default=0.3, help="m")
    p.add_argument("--hold", type=float, default=1.0, help="s inside the radius to count as reached")
    p.add_argument("--wp-timeout", type=float, default=20.0, help="s allowed to settle on a waypoint")
    p.add_argument("--max-radius", type=float, default=15.0, help="refuse plans beyond this (m)")
    p.add_argument("--max-alt", type=float, default=5.0, help="refuse altitudes above this (m)")
    p.add_argument("--min-battery", type=float, default=40.0, help="percent")
    p.add_argument("--on-abort", choices=["land", "hold"], default="land")
    p.add_argument("--no-return", action="store_true", help="land at the last waypoint")
    p.add_argument("--no-merge", action="store_true",
                   help="stop at every waypoint (default merges straight runs)")
    p.add_argument("--log", default="uav_mission_log.csv")
    p.add_argument("--plan-only", action="store_true", help="print the plan, don't fly")
    return p


def main(argv=None):
    args, _unknown = build_parser().parse_known_args(argv)  # tolerate ros2 run/launch args

    rel = load_waypoints(args.waypoints) if args.waypoints else list(DEMOS[args.demo])
    n_raw = len(rel)
    if not args.no_merge:
        rel = merge_collinear(rel)

    print("PLAN  frame=%s scale=%.2f altitude=%.1f m speed=%.2f m/s"
          % (args.frame, args.scale, args.altitude, args.speed))
    print("  %d waypoints (%d after merging straight runs)" % (n_raw, len(rel)))
    prev = (0.0, 0.0)
    total = 0.0
    for i, (x, y) in enumerate(rel, 1):
        x, y = x * args.scale, y * args.scale
        leg = math.hypot(x - prev[0], y - prev[1])
        total += leg
        print("  wp%-3d x=%7.2f  y=%7.2f   leg %.2f m" % (i, x, y, leg))
        prev = (x, y)
    ret = 0.0 if args.no_return else math.hypot(prev[0], prev[1])
    t_est = (2 * args.altitude / args.climb_speed + (total + ret) / args.speed
             + (len(rel) + 2) * (args.hold + 2.0))
    print("  path %.2f m + return %.2f m; estimated flight time ~%.0f s" % (total, ret, t_est))

    problems, _far = check_plan(rel, args.scale, args.altitude, args.max_radius, args.max_alt)
    if problems:
        for pr in problems:
            print("REFUSED: " + pr)
        return 2
    if args.plan_only:
        print("plan OK (--plan-only, not flying)")
        return 0

    m = None
    try:
        m = Mission(args)
        m.run(rel)
        return 0
    except PilotTakeover as e:
        m.warn("pilot took over (%s) -- script stopped commanding" % e)
        return 1
    except (Abort, KeyboardInterrupt) as e:
        if m is None:
            print("ABORT before flight: %s" % e)
        else:
            m.emergency(str(e) or "Ctrl-C")
        return 1
    finally:
        if m is not None:
            m.close()

if __name__ == "__main__":
    sys.exit(main())
