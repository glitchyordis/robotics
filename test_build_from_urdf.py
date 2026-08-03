import os

import pinocchio as pin
from pink.visualization import start_meshcat_visualizer
from robot_descriptions.loaders.pinocchio import load_robot_description

arm = load_robot_description("ur5e_description")
gripper_urdf = os.path.join(os.getcwd(), "hande.urdf")
package_dir = os.path.join(os.getcwd(), "src")

gripper = pin.RobotWrapper.BuildFromURDF(
    gripper_urdf,
    package_dirs=[package_dir],
)

tool0_id = arm.model.getFrameId("tool0")
placement = pin.SE3.Identity()

model, collision_model = pin.appendModel(
    arm.model, gripper.model,
    arm.collision_model, gripper.collision_model,
    tool0_id, placement
)

_, visual_model = pin.appendModel(
    arm.model, gripper.model,
    arm.visual_model, gripper.visual_model,
    tool0_id, placement
)

robot = pin.RobotWrapper(model, collision_model, visual_model)

viz = start_meshcat_visualizer(robot)
viz.display(pin.neutral(robot.model))

input("Meshcat is running. Press Enter to exit.\n")

