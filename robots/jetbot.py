"""Jetbot differential-drive robot (Isaac Lab articulation config)."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Kinematic parameters used to convert (v, w) commands into wheel speeds.
JETBOT_WHEEL_RADIUS = 0.0325  # m
JETBOT_WHEEL_BASE = 0.1125  # m, distance between the two wheels

JETBOT_WHEEL_JOINTS = ("left_wheel_joint", "right_wheel_joint")

# The USD asset limits the wheels to 12.56 rad/s (~0.41 m/s). The command range needs up to
# (0.5 + 0.5 * 0.1125 / 2) / 0.0325 ~= 16.3 rad/s on the outer wheel, so the limit is raised.
JETBOT_WHEEL_VELOCITY_LIMIT = 20.0  # rad/s

JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=100.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={
        # Velocity-controlled wheels. None = keep the drive stiffness/damping stored in the USD asset.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            stiffness=None,
            damping=None,
            velocity_limit_sim=JETBOT_WHEEL_VELOCITY_LIMIT,
        ),
    },
)
