"""Dual UR5e MuJoCo server / visualizer with two-way FastAPI communication.

see docs/dual_ur5e_server.md documentation for details.

Based on ``mujoco_dual_ur5e.py``. The MuJoCo simulation and (optionally) the
passive viewer run on the main thread, while a FastAPI app served by uvicorn
runs in a background daemon thread.

Two-way communication:
- Clients POST joint commands over REST (per robot: ``left`` / ``right``).
- Clients receive streamed state (joint positions, velocities, end-effector
  poses, optional camera frames) over a WebSocket.

Run::
    python mujoco_dual_ur5e_server.py                 # windowed viewer
    python mujoco_dual_ur5e_server.py --headless      # no window
    python mujoco_dual_ur5e_server.py --camera        # enable offscreen render
    python mujoco_dual_ur5e_server.py --physics       # dynamics via mj_step

Show or hide the joint controls at runtime with ``POST /sliders``.

Verifiable cobot position:
{
  "left": [0.82,-1.85,-1.25,-2.23,1.08,-2.16],
  "right": [-1.57,-0.52,-2.79,0.96,0.0,0.0]
}
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import threading
import time
from pathlib import Path
from typing import Optional

import mujoco.viewer
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from loop_rate_limiters import RateLimiter
from PIL import Image
from pydantic import BaseModel, Field
from robot_descriptions import ur5e_mj_description

import mujoco

BASE_BODY_NAME = "base"
EE_SITE_NAME = "attachment_site"
DOF_PER_ROBOT = 6

# Two robots placed side by side (offset along y) and inclined ~45 degrees.
ROBOT_PREFIXES = ("left_", "right_")
HOME_QPOS = np.array([0, 0, 0, 0, 0, 0.0], dtype=np.float64)
ROBOT_OFFSETS = (
    np.array([0.0, 124.3104 * 1e-3 / 2, 1], dtype=np.float64),
    np.array([0.0, -124.3104 * 1e-3 / 2, 1], dtype=np.float64),
)
INCLINE_DEG = (-45, 45)
INCLINE_AXIS = np.array([1, 0, 0.0], dtype=np.float64)
YAW_DEG = (180, 0)
YAW_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# visualization
AXIS_LENGTH = 0.12
AXIS_RADIUS = 0.006
AXIS_COLORS = (
    np.array([1.0, 0.2, 0.2, 0.9], dtype=np.float64),
    np.array([0.2, 1.0, 0.2, 0.9], dtype=np.float64),
    np.array([0.2, 0.4, 1.0, 0.9], dtype=np.float64),
)
LABEL_OFFSET = 0.06
LABEL_RGBA = np.array([0.15, 0.15, 0.15, 1.0], dtype=np.float64)

# Map friendly robot names to the internal MuJoCo prefixes.
ROBOT_NAME_TO_PREFIX = {"left": "left_", "right": "right_"}

# Offscreen camera (scaffold; disabled by default).
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_JPEG_QUALITY = 60


# --------------------------------------------------------------------------- #
# Geometry / drawing helpers (copied from mujoco_dual_ur5e.py).
# --------------------------------------------------------------------------- #
def axis_angle_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    result = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulQuat(result, q1, q2)
    return result


def add_capsule_marker(scene, start, end, radius, rgba):
    """
    adds one temporary visual marker to a MuJoCo viewer scene.
    In this case, the marker is a capsule, which is useful for drawing a thick line
    between two 3D points.

    A capsule is like a cylinder with rounded ends:
        (start) o================o (end)
        This is useful for drawing thick axes, lines, links, or direction markers.
    """
    if scene.ngeom >= scene.maxgeom:
        return

    """
    scene.geoms is an array of reusable MuJoCo visual geometry slots.
    scene.geoms[scene.ngeom] gets the next unused geometry slot. 
    Then scene.ngeom += 1 tells MuJoCo:
        This slot is now active and should be rendered.

    mental model:
        scene.geoms = [geom0, geom1, geom2, geom3, ...]
        scene.ngeom = 2

        Used:      geom0, geom1
        Next free: geom2

        So this line grabs geom2.
    """
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1

    """
    mjv_initGeom initializes a visual geometry object.
    """

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),  # shape dimensions
        np.zeros(3, dtype=np.float64),  # position
        np.zeros(9, dtype=np.float64),  # orientation matrix
        rgba,
    )

    """
    mjv_connector is a MuJoCo helper function that turns a geometry into a 
    connector between two 3D points.

    here: modifies geom so that the capsule stretches from start to end.
    
    instead of manually calculating:
        capsule center
        capsule length
        capsule orientation matrix
        MuJoCo computes those for you.
    """
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
    # MuJoCo often stores rotation matrices as flat 9-element arrays:
    # [x00, x01, x02, x10, x11, x12, x20, x21, x22]\
    # reshape to:
    # [
    #     [x00, x01, x02],
    #     [x10, x11, x12],
    #     [x20, x21, x22],
    # ]

    for axis_index, color in enumerate(AXIS_COLORS):
        axis_dir = xmat[:, axis_index]
        add_capsule_marker(
            scene,
            xpos,
            xpos + axis_length * axis_dir,
            AXIS_RADIUS,
            color,
        )


def draw_body_axes(scene, data, body_id, axis_length=AXIS_LENGTH):
    draw_frame_axes(scene, data.xpos[body_id], data.xmat[body_id], axis_length)


def draw_world_axes(scene, axis_length=AXIS_LENGTH):
    draw_frame_axes(
        scene,
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64),
        axis_length,
    )


def draw_site_axes(scene, data, site_id, axis_length=AXIS_LENGTH):
    draw_frame_axes(
        scene, data.site_xpos[site_id], data.site_xmat[site_id], axis_length
    )


def draw_site_label(scene, data, site_id, label: str, label_offset=LABEL_OFFSET):
    if scene.ngeom >= scene.maxgeom:
        return

    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1

    site_pos = data.site_xpos[site_id]
    site_xmat = data.site_xmat[site_id]
    label_pos = site_pos + label_offset * site_xmat.reshape(3, 3)[:, 2]

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LABEL,
        np.zeros(3, dtype=np.float64),
        label_pos,
        site_xmat.copy(),
        LABEL_RGBA,
    )
    geom.label = label


# --------------------------------------------------------------------------- #
# Scene assembly (copied / adapted from mujoco_dual_ur5e.py).
# --------------------------------------------------------------------------- #
def build_model() -> mujoco.MjModel:
    """Compose a scene holding two prefixed UR5e robots.
    
    cylinder: size[r, half lenght]
    cylinder_pos = ideal cylinder head start pos, (e.g. tcp[z]) + half_length
    """
    mjcf_path = Path(ur5e_mj_description.MJCF_PATH)
    world_spec = mujoco.MjSpec.from_string(
        """
<mujoco model="ur5e_dual_scene">
    <compiler angle="radian"/>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
        <geom type="cylinder" size="0.03 0.05" pos="0.8205182754378367 0.1318120808721239 0.7001129577397678" euler="0 1.57 0" rgba="0.8 0.3 0.3 1"/>
    </worldbody>
</mujoco>
"""
    )

    for prefix, offset, incline_deg, yaw_deg in zip(
        ROBOT_PREFIXES, ROBOT_OFFSETS, INCLINE_DEG, YAW_DEG
    ):
        robot_spec = mujoco.MjSpec.from_file(str(mjcf_path))

        """
        gets the root body of the robot model. In a typical robot MJCF, 
        the first body under worldbody is the robot's base body. 
        This is the body that gets attached into the larger world scene.
        """
        robot_base = robot_spec.worldbody.first_body()

        """
        A frame is like a mounting transform. 
        It defines where and how the robot should be placed relative to the world. 
        """
        frame = world_spec.worldbody.add_frame()
        frame.pos = offset

        incline_quat = axis_angle_quat(INCLINE_AXIS, np.deg2rad(incline_deg))
        yaw_quat = axis_angle_quat(YAW_AXIS, np.deg2rad(yaw_deg))
        # Post-multiply so the yaw is about the robot's own (local) z axis.
        frame.quat = quat_mul(incline_quat, yaw_quat)
        frame.attach_body(robot_base, prefix, "")

    return world_spec.compile()


# --------------------------------------------------------------------------- #
# Shared state between the sim thread and the FastAPI thread.
# --------------------------------------------------------------------------- #
class SimState:
    """Thread-safe bridge between the simulation loop and the web server."""

    def __init__(self, sliders_available: bool = True):
        self._lock = threading.Lock()

        # Pending per-robot joint commands keyed by robot name; consumed by sim.
        self._pending: dict[str, np.ndarray] = {}
        self._home_requested = False

        # Latest published snapshot (plain Python types, ready to serialize).
        self._snapshot: dict = {}

        # Camera scaffold.
        self._camera_enabled = False
        self._latest_frame_jpeg: Optional[bytes] = None

        self._sliders_available = sliders_available
        self._sliders_enabled = False
        self.shutdown = threading.Event()

    # -- command side (written by web server, read by sim) ------------------ #
    def queue_command(self, robot: str, positions: np.ndarray) -> None:
        with self._lock:
            self._pending[robot] = np.asarray(positions, dtype=np.float64).copy()

    def request_home(self) -> None:
        with self._lock:
            self._home_requested = True
            self._pending.clear()

    def take_commands(self) -> tuple[dict[str, np.ndarray], bool]:
        with self._lock:
            pending = self._pending
            self._pending = {}
            home = self._home_requested
            self._home_requested = False
            return pending, home

    # -- camera toggle ------------------------------------------------------ #
    def set_camera_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._camera_enabled = enabled
            if not enabled:
                self._latest_frame_jpeg = None

    @property
    def camera_enabled(self) -> bool:
        with self._lock:
            return self._camera_enabled

    # -- snapshot side (written by sim, read by web server) ----------------- #
    def publish(self, snapshot: dict, frame_jpeg: Optional[bytes]) -> None:
        with self._lock:
            self._snapshot = snapshot
            if frame_jpeg is not None:
                self._latest_frame_jpeg = frame_jpeg

    def get_snapshot(self) -> dict:
        with self._lock:
            return dict(self._snapshot)

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_frame_jpeg

    # -- manual joint controls --------------------------------------------- #
    def set_sliders_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._sliders_enabled = enabled and self._sliders_available

    @property
    def sliders_enabled(self) -> bool:
        with self._lock:
            return self._sliders_enabled

    @property
    def sliders_available(self) -> bool:
        return self._sliders_available


def _validate_positions(positions: list[float]):
    if len(positions) != DOF_PER_ROBOT:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {DOF_PER_ROBOT} joint positions, got {len(positions)}.",
        )
    return np.asarray(positions, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Pydantic request models.
# --------------------------------------------------------------------------- #
class JointCommand(BaseModel):
    robot: str = Field(..., description="Robot name: 'left' or 'right'.")
    positions: list[float] = Field(..., description="Six joint positions (rad).")


class BatchCommand(BaseModel):
    left: Optional[list[float]] = None
    right: Optional[list[float]] = None


class CameraToggle(BaseModel):
    enabled: bool


class SliderToggle(BaseModel):
    enabled: bool


async def _async_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def create_app(state: SimState) -> FastAPI:
    app = FastAPI(title="Dual UR5e MuJoCo Server")

    @app.get("/state")
    def get_state() -> dict:
        return state.get_snapshot()

    @app.post("/command")
    def post_command(cmd: JointCommand) -> dict:
        if cmd.robot not in ROBOT_NAME_TO_PREFIX:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown robot '{cmd.robot}'. Use 'left' or 'right'.",
            )
        positions = _validate_positions(cmd.positions)
        state.queue_command(cmd.robot, positions)
        return {"status": "ok", "robot": cmd.robot}

    @app.post("/command/batch")
    def post_command_batch(cmd: BatchCommand) -> dict:
        applied = []
        if cmd.left is not None:
            state.queue_command("left", _validate_positions(cmd.left))
            applied.append("left")
        if cmd.right is not None:
            state.queue_command("right", _validate_positions(cmd.right))
            applied.append("right")
        if not applied:
            raise HTTPException(status_code=422, detail="No joint targets provided.")
        return {"status": "ok", "applied": applied}

    @app.post("/home")
    def post_home() -> dict:
        state.request_home()
        return {"status": "ok"}

    @app.post("/camera")
    def post_camera(toggle: CameraToggle):
        state.set_camera_enabled(toggle.enabled)
        return {"status": "ok", "camera_enabled": toggle.enabled}

    @app.get("/sliders")
    def get_sliders() -> dict:
        return {
            "available": state.sliders_available,
            "enabled": state.sliders_enabled,
        }

    @app.post("/sliders")
    def post_sliders(toggle: SliderToggle) -> dict:
        if toggle.enabled and not state.sliders_available:
            raise HTTPException(
                status_code=409,
                detail="Joint sliders are unavailable in headless mode.",
            )
        state.set_sliders_enabled(toggle.enabled)
        return {"status": "ok", "enabled": state.sliders_enabled}

    @app.get("/camera/frame")
    def get_camera_frame() -> dict:
        if not state.camera_enabled:
            raise HTTPException(status_code=404, detail="Camera streaming is disabled.")
        frame = state.get_frame_jpeg()
        if frame is None:
            raise HTTPException(status_code=404, detail="No frame available yet.")
        return {"jpeg_base64": base64.b64encode(frame).decode("ascii")}

    @app.websocket("/ws/state")
    async def ws_state(websocket: WebSocket) -> None:
        await websocket.accept()
        rate_hz = 60.0
        try:
            while not state.shutdown.is_set():
                snapshot = state.get_snapshot()
                if state.camera_enabled:
                    frame = state.get_frame_jpeg()
                    if frame is not None:
                        snapshot = dict(snapshot)
                        snapshot["camera_jpeg_base64"] = base64.b64encode(frame).decode(
                            "ascii"
                        )
                await websocket.send_json(snapshot)
                await _async_sleep(1.0 / rate_hz)
        except WebSocketDisconnect:
            pass

    return app


# --------------------------------------------------------------------------- #
# Simulation loop.
# --------------------------------------------------------------------------- #
class Simulation:
    def __init__(
        self, state: SimState, enable_camera: bool, enable_physics: bool = False
    ):
        self.state = state
        self.physics = enable_physics
        self.model = build_model()
        self.data = mujoco.MjData(self.model)

        """
        for each robot, find the numeric ID of its base body and end-effector site, 
        then save those IDs for fast use later.
        
        “Find the body named left_base, and give me its internal numeric ID.”:
        self.model.body("left_base").id 
        
        That ID can then be used to index arrays like:
        self.data.xpos[body_id]
        self.data.xmat[body_id]
        Those arrays contain the body's current world position and orientation.
        """
        self.base_body_ids = [
            self.model.body(prefix + BASE_BODY_NAME).id for prefix in ROBOT_PREFIXES
        ]

        """
        "left_attachment_site"
        "right_attachment_site"  
        
        A MuJoCo site is a named marker frame in the model. It usually has no physical mass or collision by itself. 
        It is often used as a reference point: 
            tool center point, camera mount, grasp point, sensor location, etc.  
        """
        self.ee_site_ids = [
            self.model.site(prefix + EE_SITE_NAME).id for prefix in ROBOT_PREFIXES
        ]

        """
        MuJoCo model:
            names -> IDs

        MuJoCo data:
            IDs -> current numeric state
        """
        # This is your controller/server clock.
        # Run my control loop 200 times per second.
        # so So one loop takes: 0.005 seconds = 5 milliseconds
        # Then MuJoCo has its own clock:
        # self.model.opt.timestep. This is MuJoCo’s physics step size.
        # self.model.opt.timestep = 0.001 means one MuJoCo physics step advances: 0.001 seconds = 1 millisecond
        # self.n_substeps = max(1, round(self.dt / self.model.opt.timestep)) asks
        # How many MuJoCo physics steps fit inside one controller tick?

        # Number of physics integration steps per control tick so the simulation
        # advances ~one control period of wall time each loop iteration.
        # computes how many MuJoCo physics steps should happen during one server/control-loop tick.
        # e.g.
        # self.dt = 0.005                 # 200 Hz control loop
        # self.model.opt.timestep = 0.001 # 1000 Hz MuJoCo physics
        # self.dt / self.model.opt.timestep
        # 0.005 / 0.001 = 5
        # So self.n_substeps becomes 5. That means every time your server loop calls something like sim.step(),
        # the simulation should call mujoco.mj_step(...) about 5 times to advance approximately 0.005 seconds of
        # simulated time.
        # max(1, ...) protects against getting 0 substeps if the model timestep is larger than self.dt.
        # No matter what, the simulation will advance at least one MuJoCo step per control tick.
        # self.n_substeps = 5 means Meaning: every time your server loop runs once, MuJoCo should step 5 times.
        # so
        # server/control loop:
        # |--------- 5 ms ---------|
        # MuJoCo physics:
        # |1ms|1ms|1ms|1ms|1ms|

        # Usually you do not pick self.n_substeps directly. You pick these two things:
        # self.dt
        # self.model.opt.timestep
        # Then compute: self.n_substeps
        # self.dt = 1.0 / 200.0 means

        # our commands/state updates run at 200 Hz. This is a good control rate for robot simulation.
        # Then choose MuJoCo timestep smaller than that. Common choices:
        # 0.002  seconds = 500 Hz physics
        # 0.001  seconds = 1000 Hz physics
        # 0.0005 seconds = 2000 Hz physics

        # Rules Of Thumb
        # Use timestep=0.001 if you want a solid default for robot arms.
        # Use timestep=0.002 if simulation is too slow and you do not need very accurate contact/dynamics.
        # Use timestep=0.0005 if physics is unstable, contacts are jittery, or fast motion needs more accuracy.

        # Keep this relationship: MuJoCo timestep should usually be smaller than controller dt.

        # So this is good:
        # controller dt = 0.005
        # physics timestep = 0.001
        # n_substeps = 5
        # This is less ideal:
        # controller dt = 0.005
        # physics timestep = 0.01
        # n_substeps = 1
        # because one MuJoCo step is already larger than one controller period.
        # For your dual UR5e server, I would pick:
        # Controller rate: 200 Hz
        # MuJoCo timestep: 0.001
        # n_substeps: 5
        # That is the clean, normal answer.

        self.dt = 1.0 / 200.0
        self.n_substeps = max(1, round(self.dt / self.model.opt.timestep))

        # self.model.nq means the total number of generalized position coordinates,
        # not necessarily just robot joints. In your dual UR5e case, this is likely 12 if each robot
        # contributes 6 revolute joints and there are no free bodies or extra coordinates.
        # _prev_qpos is usually used to estimate velocity manually in kinematic mode:
        self._prev_qpos = np.zeros(self.model.nq, dtype=np.float64)

        # Offscreen renderer (scaffold). Created lazily so headless setups
        # without a GL context don't pay the cost unless camera is requested.
        self._renderer: Optional[mujoco.Renderer] = None
        if enable_camera:
            self.state.set_camera_enabled(True)

        self._reset_home()

    def _reset_home(self):
        for robot_index in range(len(ROBOT_PREFIXES)):
            """
            self.data.qpos:
            index:      0  1  2  3  4  5   6  7  8  9  10 11
            robot:      left robot           right robot
            joint:      j1 j2 j3 j4 j5 j6    j1 j2 j3 j4 j5 j6
            """
            lo = robot_index * DOF_PER_ROBOT
            self.data.qpos[lo : lo + DOF_PER_ROBOT] = HOME_QPOS

        if self.physics:
            # Zero velocities and point the position actuators at the home pose
            # so the arms hold station instead of collapsing under gravity.
            self.data.qvel[:] = 0.0
            for robot_index in range(len(ROBOT_PREFIXES)):
                lo = robot_index * DOF_PER_ROBOT
                self.data.ctrl[lo : lo + DOF_PER_ROBOT] = HOME_QPOS

        mujoco.mj_forward(self.model, self.data)
        self._prev_qpos[:] = self.data.qpos

    def _build_snapshot(self, velocity: np.ndarray) -> dict:
        robots = {}
        for robot_index, prefix in enumerate(ROBOT_PREFIXES):
            name = prefix.rstrip("_")
            lo = robot_index * DOF_PER_ROBOT
            ee_site_id = self.ee_site_ids[robot_index]
            robots[name] = {
                "positions": self.data.qpos[lo : lo + DOF_PER_ROBOT].tolist(),
                "velocities": velocity[
                    lo : lo + DOF_PER_ROBOT
                ].tolist(),  #  come from MuJoCo’s qvel; in kinematic mode your code estimates them from finite differences.
                "ee_position": self.data.site_xpos[ee_site_id].tolist(),
                "ee_xmat": self.data.site_xmat[ee_site_id].tolist(),
            }
        return {"time": time.time(), "robots": robots}

    def _ensure_renderer(self):
        if self._renderer is None:
            try:
                self._renderer = mujoco.Renderer(
                    self.model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH
                )
            except Exception as exc:  # pragma: no cover - depends on GL backend
                print(f"[server] Failed to create offscreen renderer: {exc}")
                self.state.set_camera_enabled(False)
                return None
        return self._renderer

    def _render_frame(self) -> Optional[bytes]:
        renderer = self._ensure_renderer()
        if renderer is None:
            return None

        renderer.update_scene(self.data)
        pixels = renderer.render()  # (H, W, 3) uint8

        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, format="JPEG", quality=CAMERA_JPEG_QUALITY)
        return buf.getvalue()

    def _apply_commands(self) -> None:
        pending, home = self.state.take_commands()
        if home:
            self._reset_home()

        for robot_name, positions in pending.items():
            prefix = ROBOT_NAME_TO_PREFIX[robot_name]
            robot_index = ROBOT_PREFIXES.index(prefix)
            lo = robot_index * DOF_PER_ROBOT
            if self.physics:
                # Drive position actuators toward the target; mj_step integrates.
                self.data.ctrl[lo : lo + DOF_PER_ROBOT] = positions
            else:
                self.data.qpos[lo : lo + DOF_PER_ROBOT] = positions

        if not self.physics and (pending or home):
            # call mj_forward so the visualized robot state updates immediately.
            mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        """
        does roughly:
        1. Read queued commands from the FastAPI server.
        2. Apply joint targets.
        3. If physics mode is enabled, call mj_step several times.
        4. If kinematic mode is enabled, update qpos directly and call mj_forward.
        5. Compute/publish robot state.
        6. Optionally render camera frames.
        """
        self._apply_commands()
        if self.physics:
            # Integrate dynamics (gravity, inertia, contacts, actuators).
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)
            velocity = self.data.qvel.copy()
        else:
            # Kinematic control: derive joint velocity by finite differencing qpos.
            velocity = (self.data.qpos - self._prev_qpos) / self.dt

        self._prev_qpos[:] = self.data.qpos
        frame_jpeg = None
        if self.state.camera_enabled:
            frame_jpeg = self._render_frame()

        self.state.publish(self._build_snapshot(velocity), frame_jpeg)

    def draw(self, scene) -> None:
        """
        scene.ngeom is the count of temproary geoms drawn like
        capsules afor axes or labesl. get's reset to 0 before redrawing.
        """
        scene.ngeom = 0
        draw_world_axes(scene)
        for robot_index, (base_body_id, ee_site_id) in enumerate(
            zip(self.base_body_ids, self.ee_site_ids)
        ):
            draw_body_axes(scene, self.data, base_body_id)
            draw_site_axes(scene, self.data, ee_site_id)
            draw_site_label(
                scene,
                self.data,
                ee_site_id,
                f"{ROBOT_PREFIXES[robot_index].rstrip('_')} tcp",
            )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

class ManualJointPanel:
    """Joint sliders and numeric inputs for passive-viewer operation."""

    def __init__(self, sim: Simulation, state: SimState):
        import tkinter as tk

        self._tk = tk
        self._sim = sim
        self._state = state
        self._root = tk.Tk()
        self._root.title("Dual UR5e Joint Control")
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._closed = False
        self._updating_unit = False
        self._controls = []
        self._scale_vars = {}
        self._active_dofs = set()
        
        self._targets = {
            prefix.rstrip("_"): sim.data.qpos[
                robot_index * DOF_PER_ROBOT : (robot_index + 1) * DOF_PER_ROBOT
            ].copy()
            for robot_index, prefix in enumerate(ROBOT_PREFIXES)
        }

        self._unit = tk.StringVar(value="rad")
        unit_frame = tk.Frame(self._root)
        unit_frame.grid(row=0, column=0, columnspan=2, pady=(8, 0))
        tk.Label(unit_frame, text="Angle unit:").pack(side=tk.LEFT, padx=(0, 6))
        tk.Radiobutton(
            unit_frame,
            text="Radians",
            variable=self._unit,
            value="rad",
            command=self._unit_changed,
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            unit_frame,
            text="Degrees",
            variable=self._unit,
            value="deg",
            command=self._unit_changed,
        ).pack(side=tk.LEFT)
        self._unit_label = tk.StringVar(value="Angle (rad)")

        for robot_index, prefix in enumerate(ROBOT_PREFIXES):
            robot_name = prefix.rstrip("_")
            frame = tk.LabelFrame(
                self._root,
                text=robot_name.capitalize(),
                padx=8,
                pady=8,
            )
            frame.grid(row=1, column=robot_index, padx=8, pady=8, sticky="n")

            tk.Label(frame, textvariable=self._unit_label).grid(
                row=0, column=1, padx=(8, 0), sticky="w"
            )

            lo = robot_index * DOF_PER_ROBOT
            for joint_index in range(DOF_PER_ROBOT):
                qpos_index = lo + joint_index
                joint = sim.model.joint(qpos_index)
                if sim.model.jnt_limited[joint.id]:
                    lower, upper = sim.model.jnt_range[joint.id]
                else:
                    lower, upper = -2.0 * np.pi, 2.0 * np.pi

                label = joint.name.removeprefix(prefix).removesuffix("_joint")
                scale_var = tk.DoubleVar(value=float(sim.data.qpos[qpos_index]))
                scale = tk.Scale(
                    frame,
                    label=label,
                    from_=float(lower),
                    to=float(upper),
                    resolution=0.01,
                    orient=tk.HORIZONTAL,
                    length=300,
                    variable=scale_var,
                )
                scale.grid(row=joint_index + 1, column=0, sticky="ew")
                scale.bind(
                    "<ButtonPress-1>",
                    lambda _event, key=(robot_name, joint_index):
                    self._active_dofs.add(key),
                )
                scale.bind(
                    "<ButtonRelease-1>",
                    lambda _event, key=(robot_name, joint_index):
                    self._active_dofs.discard(key),
                )
                self._scale_vars[(robot_name, joint_index)] = scale_var

                angle_var = tk.StringVar(value=f"{sim.data.qpos[qpos_index]:.4f}")
                entry = tk.Entry(frame, textvariable=angle_var, width=10)
                entry.grid(
                    row=joint_index + 1,
                    column=1,
                    padx=(8, 0),
                    sticky="w",
                )
                def commit(
                    _event,
                    slider=scale,
                    variable=angle_var,
                    name=robot_name,
                    index=joint_index,
                ):
                    self._commit_entry(slider, variable, name, index)

                entry.bind("<Return>", commit)
                entry.bind("<FocusOut>", commit)

                scale.configure(
                    command=lambda value,
                    name=robot_name,
                    index=joint_index,
                    variable=angle_var: self._slider_changed(
                        name, index, value, variable
                    )
                )
                self._controls.append(
                    (
                        scale,
                        angle_var,
                        robot_name,
                        joint_index,
                        float(lower),
                        float(upper),
                    )
                )

    def _to_display(self, radians: float) -> float:
        if self._unit.get() == "deg":
            return float(np.rad2deg(radians))
        return radians

    def _to_radians(self, displayed_value: float) -> float:
        if self._unit.get() == "deg":
            return float(np.deg2rad(displayed_value))
        return displayed_value

    def _format_angle(self, value: float) -> str:
        precision = 2 if self._unit.get() == "deg" else 4
        return f"{value:.{precision}f}"

    def _unit_changed(self) -> None:
        self._updating_unit = True
        try:
            suffix = "deg" if self._unit.get() == "deg" else "rad"
            self._unit_label.set(f"Angle ({suffix})")
            resolution = 0.1 if suffix == "deg" else 0.01
            for scale, angle_var, robot_name, joint_index, lower, upper in (
                self._controls
            ):
                displayed_value = self._to_display(
                    float(self._targets[robot_name][joint_index])
                )
                scale.configure(
                    from_=self._to_display(lower),
                    to=self._to_display(upper),
                    resolution=resolution,
                )
                self._scale_vars[(robot_name, joint_index)].set(displayed_value)
                angle_var.set(self._format_angle(displayed_value))
        finally:
            self._updating_unit = False

    def _slider_changed(
        self, robot_name: str, joint_index: int, value: str, angle_var
    ) -> None:
        displayed_value = float(value)
        angle_var.set(self._format_angle(displayed_value))
        if (
            not self._updating_unit
            and (robot_name, joint_index) in self._active_dofs
        ):
            self._queue_joint(
                robot_name, joint_index, self._to_radians(displayed_value)
            )

    def _commit_entry(
        self, scale, angle_var, robot_name: str, joint_index: int
    ) -> None:
        try:
            value = float(angle_var.get())
        except ValueError:
            angle_var.set(self._format_angle(float(scale.get())))
            return

        value = min(max(value, float(scale.cget("from"))), float(scale.cget("to")))
        self._scale_vars[(robot_name, joint_index)].set(value)
        angle_var.set(self._format_angle(value))
        self._queue_joint(robot_name, joint_index, self._to_radians(value))

    def _queue_joint(self, robot_name: str, joint_index: int, value: float) -> None:
        self._targets[robot_name][joint_index] = float(value)
        self._state.queue_command(robot_name, self._targets[robot_name])

    def update(self) -> None:
        if self._closed:
            return
        try:
            robots = self._state.get_snapshot().get("robots", {})
            self._updating_unit = True
            try:
                for scale, angle_var, robot_name, joint_index, _lower, _upper in (
                    self._controls
                ):
                    control_key = (robot_name, joint_index)
                    if control_key in self._active_dofs:
                        continue
                    positions = robots.get(robot_name, {}).get("positions")
                    if positions is None:
                        continue
                    target = float(positions[joint_index])
                    if abs(float(self._targets[robot_name][joint_index]) - target) <= 1e-9:
                        continue
                    self._targets[robot_name][joint_index] = target
                    displayed_value = self._to_display(target)
                    self._scale_vars[control_key].set(displayed_value)
                    angle_var.set(self._format_angle(displayed_value))
            finally:
                self._updating_unit = False
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self._closed = True
            self._state.set_sliders_enabled(False)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._state.set_sliders_enabled(False)
        self._root.destroy()


def run_windowed(sim: Simulation, state: SimState) -> None:
    """
    Important detail: launch_passive means your Python code controls the simulation loop.
    The viewer does not automatically advance physics by itself. You are responsible for calling:
        sim.step()
        viewer.sync()
    """
    rate = RateLimiter(frequency=200.0, warn=False)
    joint_panel: ManualJointPanel | None = None
    try:
        with (
            mujoco.viewer.launch_passive(
                model=sim.model,  # fixed structure of the robot/world.
                data=sim.data,  # changing runtime state: joint positions, body poses, site poses, velocities, contacts, etc.
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer
        ):
            # sets the viewer camera to MuJoCo’s default free camera.
            # In plain terms: it gives the window a reasonable initial camera view.
            mujoco.mjv_defaultFreeCamera(sim.model, viewer.cam)

            while viewer.is_running() and not state.shutdown.is_set():
                sim.step()
                if state.sliders_enabled:
                    if joint_panel is None or joint_panel.closed:
                        joint_panel = ManualJointPanel(sim, state)
                    joint_panel.update()
                elif joint_panel is not None:
                    joint_panel.close()
                    joint_panel = None
                    
                with viewer.lock():
                    # the viewer has its own internal rendering thread/state. Lock it before changing viewer
                    # drawing data so you do not modify the scene while the viewer is rendering.
                    # draws your extra visual markers into the viewer.
                    # From your code, that includes things like:
                    # world axes
                    # robot base axes
                    # end-effector/site axes
                    # TCP labels
                    # These are not physical MuJoCo objects. They are temporary viewer decorations.
                    sim.draw(viewer.user_scn)
                viewer.sync()
                rate.sleep()
    finally:
        if joint_panel is not None:
            joint_panel.close()
    state.shutdown.set()


def run_headless(sim: Simulation, state: SimState) -> None:
    rate = RateLimiter(frequency=200.0, warn=False)
    while not state.shutdown.is_set():
        sim.step()
        rate.sleep()


def main():
    parser = argparse.ArgumentParser(description="Dual UR5e MuJoCo FastAPI server.")
    parser.add_argument(
        "--headless", action="store_true", help="Run without the viewer window."
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Enable offscreen camera rendering at startup.",
    )
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Enable full dynamics (gravity, inertia, contacts) via mj_step. "
        "Commands drive position actuators instead of teleporting joints.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="FastAPI bind host.")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI bind port.")
    args = parser.parse_args()

    state = SimState(sliders_available=not args.headless)
    sim = Simulation(state, enable_camera=args.camera, enable_physics=args.physics)

    app = create_app(state)

    config = uvicorn.Config(
        app, host=args.host, port=args.port, log_level="info", ws="websockets"
    )
    server = uvicorn.Server(config)

    def serve():
        server.run()

    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()
    print(f"[server] FastAPI listening on http://{args.host}:{args.port}")
    print(
        f"[server] Mode: {'physics (mj_step)' if args.physics else 'kinematic (mj_forward)'}"
    )

    try:
        if args.headless:
            run_headless(sim, state)
        else:
            run_windowed(sim, state)
    except KeyboardInterrupt:
        pass
    finally:
        state.shutdown.set()
        server.should_exit = True
        sim.close()
        server_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
