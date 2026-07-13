"""Dual-arm inverse-kinematics client for ``mujoco_dual_ur5e_server.py``.

Solves IK for **both** UR5e arms each control tick and sends the combined joint
targets with a single ``POST /command/batch``. Runs asynchronously so it can also
stream the server's state back over the ``/ws/state`` WebSocket.

Each arm is solved independently against a standalone single-arm UR5e model whose
base sits at the origin; the target pose is expressed in that arm's **base frame**.
The server bakes in each arm's mounting transform, so the joint solutions map
directly onto ``left`` and ``right``.

Run (server must be running)::

    python mujoco_dual_ur5e_ik_client_dual.py
    python mujoco_dual_ur5e_ik_client_dual.py --left-z -0.5 --right-z -0.3
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
POS_TOL, ROT_TOL, STABLE_STEPS = 1e-3, 1e-2, 20


class ArmIK:
    """Self-contained pink IK solver for a single UR5e arm."""

    def __init__(self, name: str, z_offset: float, solver: str) -> None:
        self.name = name
        self.solver = solver
        self.robot = load_robot_description("ur5e_description")

        self.ee_task = FrameTask(
            "tool0", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0
        )
        self.posture_task = PostureTask(cost=1e-3)
        self.tasks = [self.ee_task, self.posture_task]

        q_ref = custom_configuration_vector(
            self.robot,
            **dict(zip(JOINT_NAMES, (math.radians(a) for a in Q_START_DEG))),
        )
        self.configuration = pink.Configuration(
            self.robot.model, self.robot.data, q_ref
        )
        for task in self.tasks:
            task.set_target_from_configuration(self.configuration)

        # Target pose in the arm base frame -> world.
        X_W_B = self.configuration.get_transform_frame_to_world("base")
        target_in_base = X_W_B.inverse() * self.ee_task.transform_target_to_world
        target_in_base.translation[2] = z_offset
        self.ee_task.set_target(X_W_B * target_in_base)

        self.ok_count = 0

    def step(self, dt: float) -> list[float]:
        """Advance IK one tick and return the new joint configuration."""
        velocity = solve_ik(self.configuration, self.tasks, dt, solver=self.solver)
        self.configuration.integrate_inplace(velocity, dt)
        return self.configuration.q.tolist()

    @property
    def converged(self) -> bool:
        current = self.configuration.get_transform_frame_to_world("tool0")
        target = self.ee_task.transform_target_to_world
        pos_err = np.linalg.norm(target.translation - current.translation)
        rot_err = np.linalg.norm(pin.log3(target.rotation.T @ current.rotation))
        self.ok_count = (
            self.ok_count + 1 if (pos_err < POS_TOL and rot_err < ROT_TOL) else 0
        )
        return self.ok_count >= STABLE_STEPS


async def ik_driver(args: argparse.Namespace, stop: asyncio.Event) -> None:
    """Solve IK for both arms and stream batched targets to the server."""
    solver = (
        "daqp"
        if "daqp" in qpsolvers.available_solvers
        else qpsolvers.available_solvers[0]
    )
    left = ArmIK("left", args.left_z, solver)
    right = ArmIK("right", args.right_z, solver)

    dt = 1.0 / CONTROL_HZ
    base_url = f"http://{args.host}:{args.port}"
    elapsed = 0.0

    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
        print(f"[ik-dual] Driving both arms on {base_url}")
        try:
            while not stop.is_set() and elapsed < args.max_seconds:
                left_q = left.step(dt)
                right_q = right.step(dt)

                # One request commands both arms at once.
                await client.post(
                    "/command/batch", json={"left": left_q, "right": right_q}
                )

                if left.converged and right.converged:
                    print("[ik-dual] Both arms reached their targets.")
                    break

                await asyncio.sleep(dt)
                elapsed += dt
            else:
                if elapsed >= args.max_seconds:
                    print("[ik-dual] Stopped: max-seconds reached.")
        finally:
            stop.set()


async def state_listener(args: argparse.Namespace, stop: asyncio.Event) -> None:
    """Print joint state streamed back from the server over the WebSocket."""
    url = f"ws://{args.host}:{args.port}/ws/state"
    try:
        async with websockets.connect(url) as ws:
            print(f"[ik-dual] WebSocket connected: {url}")
            while not stop.is_set():
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                snapshot = json.loads(message)
                robots = snapshot.get("robots", {})
                summary = " | ".join(
                    f"{name}: ee=({info['ee_position'][0]:+.3f},"
                    f"{info['ee_position'][1]:+.3f},"
                    f"{info['ee_position'][2]:+.3f})"
                    for name, info in robots.items()
                )
                print(f"[state] {summary}")
    except (OSError, websockets.exceptions.WebSocketException) as exc:
        print(f"[ik-dual] WebSocket error: {exc}")


async def run(args: argparse.Namespace) -> None:
    stop = asyncio.Event()
    await asyncio.gather(
        ik_driver(args, stop),
        state_listener(args, stop),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-arm IK client.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument(
        "--left-z",
        type=float,
        default=-0.5,
        help="Left EE target height (m) in the left base frame.",
    )
    parser.add_argument(
        "--right-z",
        type=float,
        default=-0.5,
        help="Right EE target height (m) in the right base frame.",
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
