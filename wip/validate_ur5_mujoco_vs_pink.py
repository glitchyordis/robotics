from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

HERE = Path(__file__).parent
MJCF_PATH = HERE / "mink" / "examples" / "universal_robots_ur5e" / "scene.xml"

PINK_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

MUJOCO_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


def load_pink_model():
    return load_robot_description("ur5_official_description")


def mujoco_joint_summary(model: mujoco.MjModel) -> None:
    print("MuJoCo joints:")
    for joint_name in MUJOCO_JOINTS:
        jid = model.joint(joint_name).id
        axis = model.jnt_axis[jid].copy()
        print(f"  {joint_name:14s} axis={axis} type={int(model.jnt_type[jid])}")


def pink_joint_summary(robot) -> None:
    print("Pink joints:")
    for joint_name in PINK_JOINTS:
        jid = robot.model.getJointId(joint_name)
        joint = robot.model.joints[jid]
        print(
            f"  {joint_name:18s} id={jid} nq={joint.nq} nv={joint.nv} idx_q={joint.idx_q} idx_v={joint.idx_v}"
        )


def compare_joint_order(robot, model: mujoco.MjModel) -> None:
    print("Joint order and names:")
    for pink_name, mj_name in zip(PINK_JOINTS, MUJOCO_JOINTS):
        print(f"  Pink: {pink_name:18s} <-> MuJoCo: {mj_name}")


def compare_zero_pose(robot, model: mujoco.MjModel) -> None:
    q0 = pin.neutral(robot.model)
    data = robot.model.createData()
    pin.forwardKinematics(robot.model, data, q0)
    pin.updateFramePlacements(robot.model, data)
    pink_tool = data.oMf[robot.model.getFrameId("tool0")]

    mj_data = mujoco.MjData(model)
    mujoco.mj_resetData(model, mj_data)
    mujoco.mj_forward(model, mj_data)
    mj_site = mj_data.site("attachment_site")

    pos_err = np.linalg.norm(pink_tool.translation - mj_site.xpos)
    rot_err = np.linalg.norm(pin.log3(pink_tool.rotation.T @ np.asarray(mj_site.xmat).reshape(3, 3)))

    print("Root-frame comparison (often non-zero when the URDF and MJCF use different base conventions):")
    print(f"  position error: {pos_err:.6e} m")
    print(f"  rotation error: {rot_err:.6e} rad")


def compare_joint_motion(robot, model: mujoco.MjModel, joint_name: str, step: float = 1e-4) -> None:
    q0 = pin.neutral(robot.model)
    pink_data_0 = robot.model.createData()
    pink_data_1 = robot.model.createData()

    joint_id = robot.model.getJointId(joint_name)
    joint = robot.model.joints[joint_id]
    q1 = q0.copy()
    q1[joint.idx_q] += step

    pin.forwardKinematics(robot.model, pink_data_0, q0)
    pin.forwardKinematics(robot.model, pink_data_1, q1)
    pin.updateFramePlacements(robot.model, pink_data_0)
    pin.updateFramePlacements(robot.model, pink_data_1)

    pink_tool_0 = pink_data_0.oMf[robot.model.getFrameId("tool0")]
    pink_tool_1 = pink_data_1.oMf[robot.model.getFrameId("tool0")]

    mj_data_0 = mujoco.MjData(model)
    mj_data_1 = mujoco.MjData(model)
    mujoco.mj_resetData(model, mj_data_0)
    mujoco.mj_resetData(model, mj_data_1)

    mj_joint_name = joint_name.replace("_joint", "")
    mj_joint = model.joint(mj_joint_name)
    mj_data_1.qpos[int(mj_joint.qposadr[0])] += step
    mujoco.mj_forward(model, mj_data_0)
    mujoco.mj_forward(model, mj_data_1)

    mj_site_0 = mj_data_0.site("attachment_site")
    mj_site_1 = mj_data_1.site("attachment_site")

    pink_delta = pin.log3(pink_tool_0.rotation.T @ pink_tool_1.rotation) / step
    mj_delta = pin.log3(np.asarray(mj_site_0.xmat).reshape(3, 3).T @ np.asarray(mj_site_1.xmat).reshape(3, 3)) / step
    alignment = float(np.dot(pink_delta, mj_delta) / (np.linalg.norm(pink_delta) * np.linalg.norm(mj_delta)))

    print(f"Motion check for {joint_name}:")
    print(f"  Pink angular delta vector: {pink_delta}")
    print(f"  MuJoCo angular delta vector: {mj_delta}")
    print(f"  Direction alignment: {alignment:.6f} (1.0 is ideal, -1.0 is flipped)")
    print(f"  Expected MuJoCo joint axis: {mj_joint.axis}")


def main() -> None:
    robot = load_pink_model()
    model = mujoco.MjModel.from_xml_path(MJCF_PATH.as_posix())

    pink_joint_summary(robot)
    mujoco_joint_summary(model)
    compare_joint_order(robot, model)
    compare_zero_pose(robot, model)

    print("Joint motion checks:")
    for joint_name in PINK_JOINTS:
        compare_joint_motion(robot, model, joint_name)


if __name__ == "__main__":
    main()