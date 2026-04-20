import math
import socket
import struct

import keyboard
import mujoco
import numpy as np
import pink
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.utils import custom_configuration_vector
from robot_descriptions.loaders.pinocchio import load_robot_description

UDP_HOST = "127.0.0.1"
UDP_PORT = 5005
PACKET_FLOATS = 13  # 6 joint values + 7 pose values (wxyz + xyz)

robot_joint_pos = {
    "top": (
        math.radians(-90.17),
        math.radians(-105.66),
        math.radians(-63),
        math.radians(-101.35),
        math.radians(90),
        math.radians(0),
    ),
}

joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

robot = load_robot_description("ur5e_description")


def set_target_0(configuration, end_effector_task):
    X_W_B = configuration.get_transform_frame_to_world("base")
    target_wrt_base = X_W_B.inverse() * end_effector_task.transform_target_to_world
    target_wrt_base.translation[2] = -0.5
    target_wrt_world = X_W_B * target_wrt_base
    end_effector_task.set_target(target_wrt_world)
    return target_wrt_world


def pack_state(q: np.ndarray, target_pose: pin.SE3) -> bytes:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, target_pose.rotation.reshape(9))
    pose = np.concatenate((quat, target_pose.translation))
    packet = np.concatenate((np.asarray(q, dtype=np.float64), pose))
    if packet.shape != (PACKET_FLOATS,):
        raise ValueError(f"Expected {PACKET_FLOATS} floats but got {packet.shape[0]}")
    return struct.pack(f"<{PACKET_FLOATS}d", *packet)


def main():
    end_effector_task = FrameTask(
        "tool0",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = PostureTask(cost=1e-3)
    tasks = [end_effector_task, posture_task]

    q_kwargs = dict(zip(joint_names, robot_joint_pos["top"]))
    q_ref = custom_configuration_vector(robot, **q_kwargs)
    configuration = pink.Configuration(robot.model, robot.data, q_ref)
    for task in tasks:
        task.set_target_from_configuration(configuration)

    solver = (
        "daqp" if "daqp" in qpsolvers.available_solvers else qpsolvers.available_solvers[0]
    )

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    pos_tol = 1e-3
    rot_tol = 1e-2
    stable_steps = 20

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target_addr = (UDP_HOST, UDP_PORT)
    set_target_0(configuration, end_effector_task)

    ok_count = 0
    while True:
        if keyboard.is_pressed("Esc"):
            print("Exiting loop.")
            break

        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        current_pose = configuration.get_transform_frame_to_world("tool0")
        target_pose = end_effector_task.transform_target_to_world
        pos_err = np.linalg.norm(target_pose.translation - current_pose.translation)
        rot_err = np.linalg.norm(pin.log3(target_pose.rotation.T @ current_pose.rotation))

        if pos_err < pos_tol and rot_err < rot_tol:
            ok_count += 1
        else:
            ok_count = 0

        sender.sendto(pack_state(configuration.q, target_pose), target_addr)

        if ok_count >= stable_steps:
            print(f"Target reached. pos_err={pos_err:.6f} m, rot_err={rot_err:.6f} rad")
            break

        rate.sleep()


if __name__ == "__main__":
    main()