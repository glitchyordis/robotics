"""
modified wrt.mujoco_load_ur5e_with_base.py
"""

import socket
import struct
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
from robot_descriptions import ur5e_mj_description

UDP_HOST = "127.0.0.1"
UDP_PORT = 5005
PACKET_FLOATS = 13  # 6 q values + 7 target pose values (wxyz + xyz)
PACKET_SIZE = struct.calcsize(f"<{PACKET_FLOATS}d")
BASE_BODY_NAME = "base"
EE_SITE_NAME = "attachment_site"
# Two robots placed side by side (offset along y) and inclined ~45 degrees.
ROBOT_PREFIXES = ("left_", "right_")
ROBOT_OFFSETS = (
    np.array([0.0, 0.45, 0.0], dtype=np.float64),
    np.array([0.0, -0.45, 0.0], dtype=np.float64),
)
INCLINE_DEG = (-45.0, 45.0)
INCLINE_AXIS = np.array([1, 0, 0.0], dtype=np.float64)
HOME_QPOS = np.array(
    [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float64
)
AXIS_LENGTH = 0.12
AXIS_RADIUS = 0.006
AXIS_COLORS = (
    np.array([1.0, 0.2, 0.2, 0.9], dtype=np.float64),
    np.array([0.2, 1.0, 0.2, 0.9], dtype=np.float64),
    np.array([0.2, 0.4, 1.0, 0.9], dtype=np.float64),
)


def unpack_packet(packet: bytes) -> tuple[np.ndarray, np.ndarray]:
    values = struct.unpack(f"<{PACKET_FLOATS}d", packet)
    q = np.array(values[:6], dtype=np.float64)
    target = np.array(values[6:], dtype=np.float64)
    return q, target


def axis_angle_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)


def add_capsule_marker(scene, start, end, radius, rgba):
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
        end,
    )


def draw_frame_axes(scene, xpos, xmat, axis_length=AXIS_LENGTH):
    xpos = xpos.copy()
    xmat = xmat.reshape(3, 3).copy()

    for axis_index, color in enumerate(AXIS_COLORS):
        axis_dir = xmat[:, axis_index]
        add_capsule_marker(
            scene,
            xpos,
            xpos + axis_length * axis_dir,
            AXIS_RADIUS,
            color,
        )


def draw_site_axes(scene, data, site_id, axis_length=AXIS_LENGTH):
    draw_frame_axes(
        scene, data.site_xpos[site_id], data.site_xmat[site_id], axis_length
    )


def draw_body_axes(scene, data, body_id, axis_length=AXIS_LENGTH):
    draw_frame_axes(scene, data.xpos[body_id], data.xmat[body_id], axis_length)


# Compose a scene that holds two UR5e robots, each instantiated with a unique
# prefix, placed side by side and inclined ~45 degrees.
mjcf_path = Path(ur5e_mj_description.MJCF_PATH)

world_spec = mujoco.MjSpec.from_string(
    """
<mujoco model="ur5e_dual_scene">
    <compiler angle="radian"/>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
    </worldbody>
</mujoco>
"""
)

for prefix, offset, incline_deg in zip(ROBOT_PREFIXES, ROBOT_OFFSETS, INCLINE_DEG):
    robot_spec = mujoco.MjSpec.from_file(str(mjcf_path))
    robot_base = robot_spec.worldbody.first_body()
    frame = world_spec.worldbody.add_frame()
    frame.pos = offset
    frame.quat = axis_angle_quat(INCLINE_AXIS, np.deg2rad(incline_deg))
    frame.attach_body(robot_base, prefix, "")

model = world_spec.compile()
data = mujoco.MjData(model)
base_body_ids = [model.body(prefix + BASE_BODY_NAME).id for prefix in ROBOT_PREFIXES]
ee_site_ids = [model.site(prefix + EE_SITE_NAME).id for prefix in ROBOT_PREFIXES]
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_HOST, UDP_PORT))
sock.setblocking(False)

for robot_index in range(len(ROBOT_PREFIXES)):
    data.qpos[robot_index * 6 : robot_index * 6 + 6] = HOME_QPOS
mujoco.mj_forward(model, data)

rate = RateLimiter(frequency=200.0, warn=False)

try:
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
                for robot_index in range(len(ROBOT_PREFIXES)):
                    data.qpos[robot_index * 6 : robot_index * 6 + 6] = q
                mujoco.mj_forward(model, data)

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                for base_body_id, ee_site_id in zip(base_body_ids, ee_site_ids):
                    draw_body_axes(viewer.user_scn, data, base_body_id)
                    draw_site_axes(viewer.user_scn, data, ee_site_id)
            viewer.sync()
            rate.sleep()
finally:
    sock.close()
