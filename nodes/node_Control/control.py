#!/usr/bin/env python

import rclpy
import random
from rclpy.node import Node
from collections import deque

import math
import os

import numpy as np
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy
from coord_transform import ned_to_world, ned_to_world_pose, yaw_from_quaternion, wrap_to_pi

from .cbf_filter import CBFConfig, CBFSafetyFilter, blocked_cell_centres, resolve_peers

# Occupancy snapshot written by exec (map_gen.run_segmentation) and also read by
# node_Path_Translator — same path traversal, so the two never disagree about which grid is current.
# The safety filter uses the `infl` layer, i.e. the very grid A* plans on.
_OCC_FILE = os.path.normpath(os.path.join(
    get_package_share_directory('coplan_vlm'), '..', '..', '..', '..',
    'debug', 'coplan_vlm_occupancy.npz'))

# ── Tunable defaults ─────────────────────────────────────────────────────────────────────
# The controller's tunable knobs, surfaced here for easy adjustment. Each is also exposed as a
# ROS parameter (same name, lower-case) so it can be overridden from the launch file or at
# runtime (`ros2 param set ...`) without editing this file.
DEFAULT_MAX_LINEAR_VEL = 0.15     # m/s   — forward speed clamp (conservative for bring-up)
DEFAULT_MAX_ANGULAR_VEL = 0.33    # rad/s — turn-rate clamp (conservative for bring-up)
DEFAULT_HEADING_GATE_DEG = 50.0   # deg   — only drive forward once heading error is within this
# rad/s per rad of heading error. Saturates max_angular_vel at |yaw_error| = 0.3/1.5 = 0.2 rad
# (11.5 deg), so behaviour above that angle is unchanged and the gain only shapes the final
# approach. Below it the heading settles first-order with time constant 1/1.5 = 0.67 s; the
# discrete step gain is 1.5 * 0.1 s = 0.15, far from the 2/dt oscillation limit.
DEFAULT_KP_YAW = 1.5             # rad/s per rad

# ── CBF safety-filter defaults ───────────────────────────────────────────────────────────
# Deliberately NOT restated here. Every filter knob is defined once, in CBFConfig (cbf_filter.py),
# next to the maths that justifies it; the parameter declarations below seed themselves from this
# instance. Duplicating the numbers is how they drift — an edit to one copy silently does nothing.
# Edit CBFConfig to change a default; override the ROS parameter to change one at runtime.
DEFAULT_ENABLE_SAFETY_FILTER = True   # node policy (arm/disarm), not filter tuning — so it lives here
_CBF_DEFAULTS = CBFConfig()


class ControlNode(Node):
    """Contains node to move turtlebot from the specified location into the parking space."""

    def __init__(self):
        """Attributes for the CarPark class; including sub, pub"""
        super().__init__("turtle_car")
        # Variable to hold boolean for if goal position has been reached.
        self.parked = True
        # Latest pose in the WORLD frame as (x, y, yaw), or None until the first message arrives.
        # NOT a zero-initialised PoseStamped(): its defaults are position (0,0,0) / identity
        # orientation, which is a LEGAL pose (the world origin is the camera nadir, mid-workspace),
        # so a missing pose publisher would be indistinguishable from genuinely sitting there — the
        # robot would drive, and filter, from a fiction. None makes that state explicit.
        self.world_pose = None
        # Initialize the current yaw variable to "None".
        self.current_yaw = None

        self.waypoint_queue = deque()

        self.goal_coordinates = None
        self.goal_yaw = None

        # The robot's pose always arrives in NED (real MoCap, or sim's odom_to_pose which now
        # emits NED). The controller converts NED -> world and runs the PID in the world frame,
        # where +yaw is CCW and matches the robot's angular.z — no per-frame switch or sign hack.

        # kP constant value. kP_pos is m/s per metre of position error.
        self.kP_pos = 0.75

        # Tunable knobs (defaults live at the top of this file). All are ROS parameters, so they
        # can be overridden from the launch file or at runtime without a rebuild.
        # - max_linear_vel / max_angular_vel: conservative output clamps applied before publishing.
        # - heading_gate_deg: only drive forward once |heading error| is within this angle.
        # - kp_yaw: heading gain, rad/s per rad (see DEFAULT_KP_YAW).
        self.declare_parameter("max_linear_vel", DEFAULT_MAX_LINEAR_VEL)      # m/s
        self.declare_parameter("max_angular_vel", DEFAULT_MAX_ANGULAR_VEL)    # rad/s
        self.declare_parameter("heading_gate_deg", DEFAULT_HEADING_GATE_DEG)  # deg
        self.declare_parameter("kp_yaw", DEFAULT_KP_YAW)                      # rad/s per rad
        self.max_linear_vel = self.get_parameter("max_linear_vel").get_parameter_value().double_value
        self.max_angular_vel = self.get_parameter("max_angular_vel").get_parameter_value().double_value
        self.heading_gate_rad = math.radians(
            self.get_parameter("heading_gate_deg").get_parameter_value().double_value)
        self.kp_yaw = self.get_parameter("kp_yaw").get_parameter_value().double_value

        # ── CBF safety filter ────────────────────────────────────────────────────────────
        # robot_name identifies WHICH robot this instance drives; robot_names is the same roster
        # parameter the executive, translator and visualizer already take. Peers are derived as
        # roster-minus-self so there is one source of truth for who exists.
        self.declare_parameter("enable_safety_filter", DEFAULT_ENABLE_SAFETY_FILTER)
        self.declare_parameter("robot_name", "")
        self.declare_parameter("robot_names", ["raph", "donnie"])
        # Filter knobs: default value comes from CBFConfig, never from a number written here.
        self.declare_parameter("lookahead_distance", _CBF_DEFAULTS.lookahead)
        self.declare_parameter("cbf_gamma_robot", _CBF_DEFAULTS.gamma_robot)
        self.declare_parameter("cbf_gamma_obstacle", _CBF_DEFAULTS.gamma_obstacle)
        self.declare_parameter("robot_radius_self", _CBF_DEFAULTS.radius_self)
        self.declare_parameter("robot_radius_other", _CBF_DEFAULTS.radius_other)
        self.declare_parameter("robot_robot_safety_margin", _CBF_DEFAULTS.robot_margin)
        self.declare_parameter("obstacle_safety_margin", _CBF_DEFAULTS.obstacle_margin)
        self.declare_parameter("obstacle_query_radius", _CBF_DEFAULTS.query_radius)
        self.declare_parameter("max_obstacle_constraints", _CBF_DEFAULTS.max_obstacles)
        self.declare_parameter("obstacle_downsample_resolution", _CBF_DEFAULTS.downsample)
        self.declare_parameter("solver_tolerance", _CBF_DEFAULTS.tol)
        self.declare_parameter("linear_velocity_min", _CBF_DEFAULTS.v_min)

        self.enable_safety_filter = self.get_parameter("enable_safety_filter").value
        self.robot_name = self.get_parameter("robot_name").get_parameter_value().string_value
        self.robot_names = [str(n) for n in self.get_parameter("robot_names").value]

        self.cbf = CBFSafetyFilter(CBFConfig(
            lookahead=self.get_parameter("lookahead_distance").value,
            gamma_robot=self.get_parameter("cbf_gamma_robot").value,
            gamma_obstacle=self.get_parameter("cbf_gamma_obstacle").value,
            radius_self=self.get_parameter("robot_radius_self").value,
            radius_other=self.get_parameter("robot_radius_other").value,
            robot_margin=self.get_parameter("robot_robot_safety_margin").value,
            obstacle_margin=self.get_parameter("obstacle_safety_margin").value,
            query_radius=self.get_parameter("obstacle_query_radius").value,
            max_obstacles=self.get_parameter("max_obstacle_constraints").value,
            downsample=self.get_parameter("obstacle_downsample_resolution").value,
            v_min=self.get_parameter("linear_velocity_min").value,
            v_max=self.max_linear_vel,
            omega_max=self.max_angular_vel,
            tol=self.get_parameter("solver_tolerance").value,
        ))

        # Live retuning: without this, every value above is read once here and `ros2 param set`
        # would silently change the parameter while the controller kept using the cached number.
        # Tuning kp_yaw / the CBF margins in sim is the main way this stack gets calibrated, so the
        # cache has to follow the parameter.
        self.add_on_set_parameters_callback(self._on_parameter_change)

        # Peer poses (world frame), keyed by name; None until each peer's first message.
        self.peers = resolve_peers(self.robot_name, self.robot_names)
        self.peer_xy = {name: None for name in self.peers}
        # Obstacle centres from the occupancy snapshot; reloaded when a new plan arrives.
        self.obstacle_points = np.empty((0, 2))
        self._occ_mtime = None

        # Holds the error between the current pose & goal pose readings
        self.pose_error = None
        self.yaw_error = None

        # Creates the command in the data type of Twist for Turtlebot hardware directions
        self.command = Twist()

        self.velocity_publisher = self.create_publisher(Twist, "/cmd_vel", 1)

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.mocap_subscriber = self.create_subscription(PoseStamped, "/pose_stamped", self.mocap_callback, qos)
        self.mocap_subscriber       # Prevent unused variable warning
        
        self.world_path_subscriber = self.create_subscription(Float32MultiArray, "/waypoint_path", self._world_path_cb, 1)
        self.world_path_subscriber       # Prevent unused variable warning

        # Peer poses come from the SAME topic everything else reads, /<name>/ned/pose_stamped,
        # published BEST_EFFORT — match that QoS as the executive/translator/visualizer do.
        for name in self.peers:
            self.create_subscription(
                PoseStamped, f"/{name}/ned/pose_stamped", self._make_peer_cb(name), qos)

        self.create_timer(0.1, self.drive_to_goal)

        if not self.enable_safety_filter:
            self.get_logger().warn("[cbf] safety filter DISABLED (enable_safety_filter:=false)")
        elif not self.robot_name:
            self.get_logger().error(
                "[cbf] robot_name is unset, so no peer can be identified — the robot-robot "
                "constraint is INACTIVE. Set it in the launch file.")
        elif self.robot_name not in self.robot_names:
            self.get_logger().error(
                f"[cbf] robot_name '{self.robot_name}' is not in robot_names {self.robot_names}; "
                "the robot-robot constraint is INACTIVE. Fix the launch file (a typo here would "
                "otherwise make the robot avoid itself and never move).")
        else:
            self.get_logger().info(
                f"[cbf] enabled for '{self.robot_name}'; peers={self.peers}; "
                f"R_robot={self.cbf.cfg.r_robot:.2f} m, R_obstacle={self.cbf.cfg.r_obstacle:.2f} m")
        self._load_occupancy()

        self.get_logger().info("✓")

    
    def _world_path_cb(self, msg: Float32MultiArray):
        # msg.data → [x1, y1, x2, y2, …]
        coords = [
            (msg.data[i], msg.data[i + 1])
            for i in range(0, len(msg.data), 2)
        ]
        self.waypoint_queue = deque(coords)  # or list(coords)
        # A new plan supersedes the old one. Drop the currently-active waypoint by re-parking so
        # the next control tick re-acquires from the new queue (whose first point is the robot's
        # current pose) instead of finishing the now-stale leg.
        self.parked = True
        # A new path implies exec has just written a fresh occupancy snapshot, so refresh the
        # obstacle set. It wrote the .npz BEFORE the (multi-second) planner VLM call, so by the
        # time a path reaches us that write finished long ago — the read is never racing it.
        self._load_occupancy()

    def _on_parameter_change(self, params):
        """Apply `ros2 param set` at runtime to the cached controller / filter knobs.

        Only the numeric tuning knobs are live; identity and topic wiring (robot_name, robot_names)
        are structural — they decide which subscriptions exist — so changing them here would not do
        what the caller expects and is deliberately ignored.
        """
        # parameter name -> where the cached copy lives
        controller = {"max_linear_vel": "max_linear_vel", "max_angular_vel": "max_angular_vel",
                      "kp_yaw": "kp_yaw", "enable_safety_filter": "enable_safety_filter"}
        cbf = {"lookahead_distance": "lookahead", "cbf_gamma_robot": "gamma_robot",
               "cbf_gamma_obstacle": "gamma_obstacle", "robot_radius_self": "radius_self",
               "robot_radius_other": "radius_other", "robot_robot_safety_margin": "robot_margin",
               "obstacle_safety_margin": "obstacle_margin", "obstacle_query_radius": "query_radius",
               "max_obstacle_constraints": "max_obstacles",
               "obstacle_downsample_resolution": "downsample", "solver_tolerance": "tol",
               "linear_velocity_min": "v_min"}
        for p in params:
            if p.name in controller:
                setattr(self, controller[p.name], p.value)
            elif p.name in cbf:
                setattr(self.cbf.cfg, cbf[p.name], p.value)
            elif p.name == "heading_gate_deg":
                self.heading_gate_rad = math.radians(p.value)
            else:
                continue
            self.get_logger().info(f"[param] {p.name} -> {p.value}")
        # The velocity clamps double as the QP's box bounds, so keep the two in step.
        self.cbf.cfg.v_max = self.max_linear_vel
        self.cbf.cfg.omega_max = self.max_angular_vel
        return SetParametersResult(successful=True)

    def _make_peer_cb(self, name: str):
        """Cache a peer's position in the world frame (its pose arrives in NED, like our own)."""
        def _cb(msg: PoseStamped) -> None:
            self.peer_xy[name] = ned_to_world(msg.pose.position.x, msg.pose.position.y)
        return _cb

    def _load_occupancy(self) -> None:
        """Refresh the static-obstacle set from exec's occupancy snapshot. Never raises.

        Uses the `infl` layer — the same pre-inflated grid A* plans on — so the filter's safe set
        matches the planner's and it never fights a correctly-tracked path. A failed or torn read
        keeps the previous obstacle set rather than silently dropping to "no obstacles", which would
        quietly disarm the map constraints.
        """
        if not self.enable_safety_filter:
            return
        try:
            mtime = os.path.getmtime(_OCC_FILE)
            if mtime == self._occ_mtime:
                return
            snap = np.load(_OCC_FILE)
            meta = {"resolution": float(snap["resolution"]),
                    "origin_x": float(snap["origin_x"]), "origin_y": float(snap["origin_y"]),
                    "width": int(snap["width"]), "height": int(snap["height"])}
            self.obstacle_points = blocked_cell_centres(snap["infl"], meta)
            self._occ_mtime = mtime
            self.get_logger().info(
                f"[cbf] occupancy snapshot loaded: {len(self.obstacle_points)} blocked cells "
                f"@ {meta['resolution']} m")
        except FileNotFoundError:
            self.get_logger().warn(
                "[cbf] no occupancy snapshot yet — map obstacles inactive until exec publishes a "
                "plan. Peer avoidance is unaffected.")
        except Exception as exc:                                   # noqa: BLE001
            self.get_logger().warn(
                f"[cbf] could not read the occupancy snapshot ({exc}); keeping the previous "
                f"{len(self.obstacle_points)} obstacle cells.")

    def mocap_callback(self, data):
        """Gets current position information from the motion capture cameras."""
        q = data.pose.orientation
        ned_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        # NED -> world once, here, so every consumer (P controller and safety filter alike) reads
        # one cached conversion instead of repeating it.
        self.world_pose = ned_to_world_pose(data.pose.position.x, data.pose.position.y, ned_yaw)

    def drive_to_goal(self):
        """Function that runs the P controller algorithm."""
        if self.world_pose is None:
            # No pose yet: we do not know where we are, so neither the heading error nor any
            # barrier is meaningful. Hold still rather than driving from a fictional pose.
            self.command.linear.x = 0.0
            self.command.angular.z = 0.0
            self.velocity_publisher.publish(self.command)
            return
        if self.parked and self.waypoint_queue:
            self.goal_coordinates = self.waypoint_queue.popleft()
            self.parked = False
        if not self.parked:
            self.position_error_calc()
            self.orientation_error_calc()
            self.publish_velocity()

    def position_error_calc(self):
        """Position error in the WORLD frame (the pose was converted in mocap_callback)."""
        # Waypoints from the translator are ALREADY world-frame, so we compare directly.
        self.robot_wx, self.robot_wy, self.current_yaw = self.world_pose

        x_difference = self.goal_coordinates[0] - self.robot_wx
        y_difference = self.goal_coordinates[1] - self.robot_wy

        self.pose_error = PoseStamped()
        self.pose_error.pose.position.x = x_difference
        self.pose_error.pose.position.y = y_difference

    def orientation_error_calc(self):
        """Heading error in the WORLD frame (z-up / +yaw CCW, matching the robot's angular.z)."""
        x_difference = self.pose_error.pose.position.x
        y_difference = self.pose_error.pose.position.y

        # Bearing to the goal in world, minus the robot's world heading (cached above).
        bearing_world = math.atan2(y_difference, x_difference)
        self.yaw_error = wrap_to_pi(bearing_world - self.current_yaw)

    def publish_velocity(self):
        """Publishes the linear and angular velocity commands to the hardware."""
        # yaw_error is already wrapped to [-pi, pi] by orientation_error_calc; kp_yaw is rad/s per
        # rad, matching angular.z's rad/s (REP-103). The previous gain multiplied a DEGREE value,
        # making the effective gain 0.5 * 180/pi = 28.6 rad/s per rad — it saturated max_angular_vel
        # at 0.6 deg of error, so the controller was bang-bang and the gain itself inert.
        self.command.angular.z = self.kp_yaw * self.yaw_error

        # Calculate and initiate forward movement of the robot.
        car = self.pose_error.pose.position
        error_distance = math.sqrt(car.x ** 2 + car.y ** 2)

        # Heading gate: only drive forward when roughly facing the goal. If the heading error
        # exceeds heading_gate_deg, command zero linear velocity and just rotate toward the goal
        # (prevents wide arcs that cut corners / clip obstacles). yaw_error is already wrapped to
        # [-pi, pi], so it reflects the true signed heading error.
        #
        # This flag also decides whether the safety filter runs at all (see below), so the two
        # regimes — pivot in place vs drive — are named once and used consistently.
        turning_in_place = abs(self.yaw_error) > self.heading_gate_rad
        if turning_in_place:
            self.command.linear.x = 0.0
        else:
            self.command.linear.x = self.kP_pos * error_distance

        # Checks if in proximity to target (capture radius). Mark parked so the next tick pulls
        # the following waypoint; if the route is finished (queue empty) command a real stop
        # (zero linear + angular) rather than coasting on the last non-zero command. Intermediate
        # waypoints are not zeroed, so the robot flows smoothly through them.
        if error_distance < 0.1:
            if not self.waypoint_queue:
                self.command.linear.x = 0.0
                self.command.angular.z = 0.0
            self.parked = True

        # Final safety clamp: cap both commands to the conservative limits before publishing,
        # so no combination of gain * error can send the robot an unsafe velocity.
        self.command.linear.x = max(-self.max_linear_vel,
                                    min(self.max_linear_vel, self.command.linear.x))
        self.command.angular.z = max(-self.max_angular_vel,
                                     min(self.max_angular_vel, self.command.angular.z))

        # CBF-QP safety filter — DRIVING REGIME ONLY, and applied to the CLAMPED nominal command.
        #
        # Skipped while turning in place because the barrier is evaluated on the look-ahead point,
        # which swings on a circle of radius `ell` as the robot rotates. That makes hdot non-zero
        # under pure rotation even though a circular robot pivoting about its own centre holds
        # every obstacle and peer distance CONSTANT — rotation cannot create a collision, so
        # constraining it is a modelling artifact. A costly one: with a barrier already violated
        # (h < 0) it blocks half of all turn directions, and since R_robot is 0.54 m the peer row
        # is violated whenever the robots are within half a metre, which can stop the pivot
        # outright. `turning_in_place` is this tick's freshly computed heading error, never a
        # previous filtered output, so braking to v = 0 cannot disable filtering on the next tick.
        #
        # Clamping first matters too: anchoring the QP objective at the pre-clamp value would
        # measure "minimum deviation" against a command the robot could never execute.
        if self.enable_safety_filter and not turning_in_place:
            self._apply_safety_filter()

        # Publish the updated velocity command values to the bot.
        self.velocity_publisher.publish(self.command)

    def _apply_safety_filter(self) -> None:
        """Replace self.command with the closest safe command. Never raises."""
        nominal = (self.command.linear.x, self.command.angular.z)
        result = self.cbf.filter(
            nominal=nominal,
            robot_pose=self.world_pose,
            peer_points=[self.peer_xy[n] for n in self.peers],
            obstacle_points=self.obstacle_points,
        )
        self.command.linear.x = result.v
        self.command.angular.z = result.omega

        # Throttled diagnostics only — this runs at 10 Hz, so unthrottled logging would flood.
        if result.fallback:
            self.get_logger().error(
                f"[cbf] SAFE STOP: {result.fallback} (failures={self.cbf.failure_count}, "
                f"obstacles={result.n_obstacles}, peer={result.peer_active})",
                throttle_duration_sec=2.0)
        elif result.modified:
            self.get_logger().info(
                f"[cbf] nominal ({nominal[0]:.3f}, {nominal[1]:.3f}) -> "
                f"({result.v:.3f}, {result.omega:.3f}) | obstacles={result.n_obstacles} "
                f"peer={result.peer_active} min_h={result.min_h:+.3f} "
                f"{result.status} {result.solve_ms:.2f} ms",
                throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)

    # Create a turtle bot instance & initiate setup (goal set)
    turtle_car = ControlNode()
    
    rclpy.spin(turtle_car)
        
    turtle_car.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
