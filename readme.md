This repo contains scripts 
- [TODO](#todo)
- [Codes](#codes)

## TODO
- [ ] write a desc. for [generic.ipynb](generic.ipynb), [test_spacemouse](test_spacemouse.py)

## Codes

### Dual UR5e servers & clients

- [mujoco_dual_ur5e_2f85_server](mujoco_dual_ur5e_2f85_server.py) — [mujoco_dual_ur5e_server](mujoco_dual_ur5e_server.py) with gripper control. See [bruno api collection](<api/bruno/MuJoCo Dual UR5e 2f85 Server>).
- [mujoco_dual_ur5e_server](mujoco_dual_ur5e_server.py) — server that loads 2 ur5e. Doc: [dual_ur5e_server.md](docs/dual_ur5e_server.md). Clients:
  - [mujoco_dual_ur5e_client](mujoco_dual_ur5e_client.py) — example that drives robot to perform sinusoidal movement
  - [mujoco_dual_ur5e_ik_client_sync](mujoco_dual_ur5e_ik_client_sync.py) — ik example on single arm
  - [mujoco_dual_ur5e_ik_client_async](mujoco_dual_ur5e_ik_client_async.py) — ik example on single arm, async
  - [mujoco_dual_ur5e_ik_client_dual](mujoco_dual_ur5e_ik_client_dual.py) — async ik on dual arm

### Loading UR5e

- [mujoco_load_ur5e_with_base](mujoco_load_ur5e_with_base.py) — loads ur5e in mujoco with a base
- [mujoco_load_ur5e](mujoco_load_ur5e.py) — loads ur5e in mujoco
- [mujoco_ur5e_ground_frames.py](mujoco_ur5e_ground_frames.py) — loads one ur5e on a ground plane and labels frame axes

### Inverse kinematics

- [mujoco_ur5e_viewer_server.py](mujoco_ur5e_viewer_server.py), [mujoco_pink_ur5e_inverse_kinematics.py](mujoco_pink_ur5e_inverse_kinematics.py) — inverse kin. on ur5e, viz with mujoco
- [pink_transformations.ipynb](pink_transformations.ipynb) — transformations using pink library
- [pink_ur5e_inverse_kinematic.ipynb](pink_ur5e_inverse_kinematic.ipynb) — inverse_kin with pink
- [pinochio_viz_urdf.ipynb](pinochio_viz_urdf.ipynb) — custom util to viz robot loaded with robot_description via pinnochio
