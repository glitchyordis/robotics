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
    draw_frame_axes(scene, data.site_xpos[site_id], data.site_xmat[site_id], axis_length)


def draw_body_axes(scene, data, body_id, axis_length=AXIS_LENGTH):
    draw_frame_axes(scene, data.xpos[body_id], data.xmat[body_id], axis_length)


# Build an asset bundle so MuJoCo can resolve meshes when loading from a string.
mjcf_path = Path(ur5e_mj_description.MJCF_PATH)
asset_root = mjcf_path.parent
assets = {
    path.relative_to(asset_root).as_posix(): path.read_bytes()
    for path in asset_root.rglob("*")
    if path.is_file()
}

scene_xml = """
<mujoco model="ur5e_scene">
    <include file="ur5e.xml"/>

    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(scene_xml, assets=assets)
data = mujoco.MjData(model)
base_body_id = model.body(BASE_BODY_NAME).id
ee_site_id = model.site(EE_SITE_NAME).id
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_HOST, UDP_PORT))
sock.setblocking(False)

mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
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
                data.qpos[:] = q
                mujoco.mj_forward(model, data)

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw_body_axes(viewer.user_scn, data, base_body_id)
                draw_site_axes(viewer.user_scn, data, ee_site_id)
            viewer.sync()
            rate.sleep()
finally:
    sock.close()