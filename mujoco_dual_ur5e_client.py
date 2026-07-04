"""Demo client for ``mujoco_dual_ur5e_server.py``.

Two-way communication:
- Sends per-robot joint commands over REST (``POST /command``).
- Receives streamed state (positions, velocities, EE poses, optional camera
  frames) over a WebSocket (``/ws/state``).

The default demo drives both robots with out-of-phase sinusoids while printing
the streamed state, then returns them home on exit.

Run (with the server already running)::

    python mujoco_dual_ur5e_client.py
    python mujoco_dual_ur5e_client.py --duration 10 --camera
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time

import httpx
import numpy as np
import websockets

DOF_PER_ROBOT = 6
HOME_QPOS = np.array(
    [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float64
)


def rest_base(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def ws_url(host: str, port: int) -> str:
    return f"ws://{host}:{port}/ws/state"


async def state_listener(
    host: str, port: int, stop: asyncio.Event, save_camera: bool
) -> None:
    """Connect to the WebSocket and print streamed state until ``stop`` is set."""
    url = ws_url(host, port)
    try:
        async with websockets.connect(url) as ws:
            print(f"[client] WebSocket connected: {url}")
            saved_frame = False
            while not stop.is_set():
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                snapshot = json.loads(message)
                robots = snapshot.get("robots", {})
                summary = " | ".join(
                    f"{name}: q0={info['positions'][0]:+.3f} "
                    f"v0={info['velocities'][0]:+.3f} "
                    f"ee=({info['ee_position'][0]:+.3f},"
                    f"{info['ee_position'][1]:+.3f},"
                    f"{info['ee_position'][2]:+.3f})"
                    for name, info in robots.items()
                )
                print(f"[state] {summary}")

                if save_camera and not saved_frame and "camera_jpeg_base64" in snapshot:
                    with open("camera_frame.jpg", "wb") as fh:
                        fh.write(base64.b64decode(snapshot["camera_jpeg_base64"]))
                    print("[client] Saved camera_frame.jpg")
                    saved_frame = True
    except (OSError, websockets.exceptions.WebSocketException) as exc:
        print(f"[client] WebSocket error: {exc}")


async def command_driver(
    host: str, port: int, stop: asyncio.Event, duration: float, camera: bool
) -> None:
    """Drive both robots with out-of-phase sinusoidal joint targets."""
    base = rest_base(host, port)
    amplitude = 0.5  # rad
    freq = 0.25  # Hz
    period = 1.0 / 50.0  # 50 Hz command rate
    start = time.time()

    async with httpx.AsyncClient(base_url=base, timeout=5.0) as client:
        if camera:
            await client.post("/camera", json={"enabled": True})
            print("[client] Requested camera streaming.")

        try:
            while not stop.is_set():
                t = time.time() - start
                if t >= duration:
                    break
                left = HOME_QPOS.copy()
                right = HOME_QPOS.copy()
                left[0] += amplitude * np.sin(2 * np.pi * freq * t)
                right[0] += amplitude * np.sin(2 * np.pi * freq * t + np.pi)
                await client.post(
                    "/command/batch",
                    json={"left": left.tolist(), "right": right.tolist()},
                )
                await asyncio.sleep(period)
        finally:
            # Return both robots home before exiting.
            try:
                await client.post("/home")
                print("[client] Sent /home.")
            except httpx.HTTPError as exc:
                print(f"[client] Failed to send /home: {exc}")
    stop.set()


async def run(args: argparse.Namespace) -> None:
    stop = asyncio.Event()
    await asyncio.gather(
        state_listener(args.host, args.port, stop, args.camera),
        command_driver(args.host, args.port, stop, args.duration, args.camera),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual UR5e MuJoCo demo client.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Seconds to run the sinusoidal demo.",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Enable camera streaming and save the first received frame.",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
