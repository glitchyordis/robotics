import time

import numpy as np
import pyspacemouse

MOTION_THRESHOLD = 0.01  # Threshold for combined motion magnitude before printing
DEAD_ZONE = 0.095  # Dead zone for very small movements
PRINT_RATE_LIMIT = 200  # Max prints per second
POLL_RATE = 200  # Polling rate of the SpaceMouse in Hz


def apply_dead_zone(values, threshold=DEAD_ZONE):
    """Apply dead zone to filter out tiny movements"""
    return np.where(np.abs(values) < threshold, 0, values)


if __name__ == "__main__":
    pyspacemouse.close()
    if not pyspacemouse.open():
        print("[spacemouse] failed to open SpaceMouse")
        exit()

    print("SpaceMouse ready! Move it around to see values. Press any button to exit.")

    last_print_time = 0

    try:
        while True:
            state = pyspacemouse.read()

            # Apply coordinate transformation and dead zone
            raw_lin = apply_dead_zone(np.array([state.x, state.y, state.z]))
            raw_ang = apply_dead_zone(np.array([state.roll, state.pitch, state.yaw]))

            # Calculate motion magnitude
            motion_magnitude = np.linalg.norm(raw_lin) + np.linalg.norm(raw_ang)

            # Rate-limited printing with higher threshold
            current_time = time.time()
            if (
                motion_magnitude > MOTION_THRESHOLD
                and current_time - last_print_time > 1.0 / PRINT_RATE_LIMIT
            ):
                print(
                    f"Linear: {raw_lin.round(3)}, Angular: {raw_ang.round(3)}, Mag: {motion_magnitude:.3f}"
                )
                last_print_time = current_time

            # Check for button press to exit
            if any(state.buttons):
                print(f"[spacemouse] buttons pressed: {state.buttons}")
                break

            time.sleep(0.005)  # 200Hz polling

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pyspacemouse.close()
