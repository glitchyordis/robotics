import math
import os

import numpy as np
import pink
import pinocchio as pin
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer
from robot_descriptions.loaders.pinocchio import load_robot_description

# # 1. Load the UR5e Arm from robot_descriptions
# arm = load_robot_description("ur5e_description")

# # 2. Load the Hand-E Gripper (Local URDF)
# # Make sure this path points to where your generated hande.urdf is
# urdf_path = "hande.urdf" 
# # This path should be the FOLDER that contains 'robotiq_hande_description'
# # so Pinocchio can find the meshes.
# mesh_dir = os.getcwd() 

# gripper = pin.RobotWrapper.BuildFromURDF(
#     urdf_path,
#     package_dirs=[mesh_dir]
# )


arm = load_robot_description("ur5e_description")
gripper = pin.RobotWrapper.BuildFromURDF("hande.urdf", package_dirs=[...])

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