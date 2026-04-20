import socket
import struct

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
from robot_descriptions.loaders.mujoco import load_robot_description

UDP_HOST = "127.0.0.1"
UDP_PORT = 5005
PACKET_FLOATS = 13  # 6 q values + 7 target pose values (wxyz + xyz)
PACKET_SIZE = struct.calcsize(f"<{PACKET_FLOATS}d")
WRIST_SITE_NAME = "attachment_site"
WRIST_JOINT_NAME = "wrist_3_joint"
AXIS_LENGTH = 0.25
AXIS_RADIUS = 0.015
ANCHOR_RADIUS = 0.025
AXIS_RGBA = np.array([1.0, 0.2, 0.2, 0.9], dtype=np.float32)
ANCHOR_RGBA = np.array([0.2, 1.0, 0.2, 0.9], dtype=np.float32)


def unpack_packet(packet: bytes) -> tuple[np.ndarray, np.ndarray]:
    values = struct.unpack(f"<{PACKET_FLOATS}d", packet)
    q = np.array(values[:6], dtype=np.float64)
    target = np.array(values[6:], dtype=np.float64)
    return q, target


def add_visual_axis(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float,
    rgba: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return

    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.zeros(9, dtype=np.float64),
        rgba,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        start,
        start + length * direction,
    )


def add_visual_sphere(
    scene: mujoco.MjvScene,
    center: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return

    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        center,
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )


def draw_wrist_axis(
    scene: mujoco.MjvScene,
    data: mujoco.MjData,
    joint_id: int,
) -> None:
    scene.ngeom = 0
    anchor = data.xanchor[joint_id].copy()
    axis = data.xaxis[joint_id].copy()
    add_visual_sphere(scene, anchor, ANCHOR_RADIUS, ANCHOR_RGBA)
    add_visual_axis(
        scene,
        anchor,
        axis,
        AXIS_LENGTH,
        AXIS_RADIUS,
        AXIS_RGBA,
    )


def main():
    model = load_robot_description("ur5e_mj_description")
    data = mujoco.MjData(model)
    model.site(WRIST_SITE_NAME)
    wrist_joint_id = model.joint(WRIST_JOINT_NAME).id

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_PORT))
    sock.setblocking(False)

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    rate = RateLimiter(frequency=200.0, warn=False)

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        while viewer.is_running():
            latest_packet = None
            while True:
                try:
                    latest_packet, _ = sock.recvfrom(PACKET_SIZE)
                except BlockingIOError:
                    break

            if latest_packet is not None and len(latest_packet) == PACKET_SIZE:
                q, _target = unpack_packet(latest_packet)
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                data.site(WRIST_SITE_NAME)

            with viewer.lock():
                draw_wrist_axis(viewer.user_scn, data, wrist_joint_id)
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()