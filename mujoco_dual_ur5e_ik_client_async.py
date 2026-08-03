"""Asynchronous inverse-kinematics client for ``mujoco_dual_ur5e_server.py``.

Same IK solve as ``mujoco_dual_ur5e_ik_client_sync.py``, but runs two coroutines
concurrently with ``asyncio``:

- ``ik_driver``  : solves IK each tick and POSTs ``/command`` (httpx.AsyncClient).
- ``state_listener``: subscribes to the ``/ws/state`` WebSocket and prints the
  joint state the server streams back (useful to watch physics-mode ``qvel``).

This is the pattern to use when you want live feedback *while* commanding the
robot. For pure command-sending the synchronous client is simpler.

Run (server must be running)::

    python mujoco_dual_ur5e_ik_client_async.py
    python mujoco_dual_ur5e_ik_client_async.py --robot right --z-offset -0.4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math

import httpx
import numpy as np
import pink
import pinocchio as pin
import qpsolvers
import websockets
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.utils import custom_configuration_vector
from robot_descriptions.loaders.pinocchio import load_robot_description

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
Q_START_DEG = (-90.17, -105.66, -63.0, -101.35, 90.0, 0.0)
CONTROL_HZ = 200.0


def build_configuration(robot) -> pink.Configuration:
    q_ref = custom_configuration_vector(
        robot, **dict(zip(JOINT_NAMES, (math.radians(a) for a in Q_START_DEG)))
    )
    return pink.Configuration(robot.model, robot.data, q_ref)


async def ik_driver(args: argparse.Namespace, stop: asyncio.Event) -> None:
    """Solve IK and stream joint targets to the server over REST."""
    robot = load_robot_description("ur5e_description")

    ee_task = FrameTask(
        "tool0", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0
    )
    posture_task = PostureTask(cost=1e-3)
    tasks = [ee_task, posture_task]

    configuration = build_configuration(robot)
    for task in tasks:
        task.set_target_from_configuration(configuration)

    solver = (
        "daqp"
        if "daqp" in qpsolvers.available_solvers
        else qpsolvers.available_solvers[0]
    )

    # Target pose in the robot base frame -> world.
    X_W_B = configuration.get_transform_frame_to_world("base")
    target_in_base = X_W_B.inverse() * ee_task.transform_target_to_world
    target_in_base.translation[2] = args.z_offset
    ee_task.set_target(X_W_B * target_in_base)

    dt = 1.0 / CONTROL_HZ
    pos_tol, rot_tol, stable_steps = 1e-3, 1e-2, 20
    base_url = f"http://{args.host}:{args.port}"
    ok_count = 0
    elapsed = 0.0

    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
        print(f"[ik-async] Driving '{args.robot}' on {base_url}")
        try:
            while not stop.is_set() and elapsed < args.max_seconds:
                velocity = solve_ik(configuration, tasks, dt, solver=solver)
                configuration.integrate_inplace(velocity, dt)

                await client.post(
                    "/command",
                    json={
                        "robot": args.robot,
                        "positions": configuration.q.tolist(),
                    },
                )

                current_pose = configuration.get_transform_frame_to_world("tool0")
                target_pose = ee_task.transform_target_to_world
                pos_err = np.linalg.norm(
                    target_pose.translation - current_pose.translation
                )
                rot_err = np.linalg.norm(
                    pin.log3(target_pose.rotation.T @ current_pose.rotation)
                )

                ok_count = (
                    ok_count + 1 if (pos_err < pos_tol and rot_err < rot_tol) else 0
                )
                if ok_count >= stable_steps:
                    print(
                        f"[ik-async] Target reached. "
                        f"pos_err={pos_err:.6f} m, rot_err={rot_err:.6f} rad"
                    )
                    break

                # Yield control at the desired rate without blocking the loop.
                await asyncio.sleep(dt)
                elapsed += dt
            else:
                if elapsed >= args.max_seconds:
                    print("[ik-async] Stopped: max-seconds reached.")
        finally:
            stop.set()


async def state_listener(args: argparse.Namespace, stop: asyncio.Event) -> None:
    """Print joint state streamed back from the server over the WebSocket."""
    url = f"ws://{args.host}:{args.port}/ws/state"
    try:
        async with websockets.connect(url) as ws:
            print(f"[ik-async] WebSocket connected: {url}")
            while not stop.is_set():
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                snapshot = json.loads(message)
                info = snapshot.get("robots", {}).get(args.robot)
                if info is None:
                    continue
                q0 = info["positions"][0]
                v0 = info["velocities"][0]
                ee = info["ee_position"]
                print(
                    f"[state] {args.robot}: q0={q0:+.3f} v0={v0:+.3f} "
                    f"ee=({ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f})"
                )
    except (OSError, websockets.exceptions.WebSocketException) as exc:
        print(f"[ik-async] WebSocket error: {exc}")


async def run(args: argparse.Namespace) -> None:
    stop = asyncio.Event()
    await asyncio.gather(
        ik_driver(args, stop),
        state_listener(args, stop),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Asynchronous IK client.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument(
        "--robot", choices=("left", "right"), default="left", help="Target robot."
    )
    parser.add_argument(
        "--z-offset",
        type=float,
        default=-0.5,
        help="EE target height (m) in the robot base frame.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=20.0,
        help="Safety timeout for the solve loop.",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
