'''
This is a rewrite of the old MatPlotLib based Robot Virtual Twin, using the library DearPyGUI.
The MatPlotLib implementation worked, but it was extremely difficult to add new features, everything broke.
DPG is easy to use, robust, and clean.

How it works:

There is a dicitonary of the current state of every variable being controlled by the control pannel

There is a class that creates the control pannel based on a set of dictionaries that list the
attributes of the dropdown menus, their UI elements, the variable name, and thier state.
When the variables are changed, the shared state dictionary is changed.

Every time interval (set by a delay), 
the state of every variable is sent to the program that uses those variables.
'''

import time
import threading
import dearpygui.dearpygui as dpg
import torch
import numpy as np


from forward_kinematics import Robot_math
from path_planning_math import PathPlannerMath
from expert_inv_kin import Oracle
from loss_math import Loss_Math
from robot_visualizer import Robot_Visualizer


# ----------------------------
# Shared State
# ----------------------------
class SharedState:
    def __init__(self):
        self.data = {
            "rand_bool": False,
            "my_val": 0.0,
        }
        self.lock = threading.Lock()

    def set(self, key, value):
        with self.lock:
            self.data[key] = value

    def get(self, key):
        with self.lock:
            return self.data[key]


# ----------------------------
# UI Generator with Groups
# ----------------------------
class LiveUI:
    def __init__(self, state, config):
        self.state = state
        self.config = config

    def build(self):
        with dpg.window(label="Control Panel", width=400, height=600):
            for group in self.config:
                self._create_group(group)

    def _create_group(self, group):
        label = group["group"]
        items = group["items"]

        with dpg.collapsing_header(label=label, default_open=True):
            for item in items:
                self._create_widget(item)

    def _create_widget(self, item):
        name = item["name"]
        default = item.get("default", 0)
        self.state.set(name, default)

        def callback(sender, app_data, user_data):
            self.state.set(user_data, app_data)

        if item["type"] == "float":
            dpg.add_slider_float(
                label=name,
                default_value=default,
                min_value=item.get("min", 0.0),
                max_value=item.get("max", 1.0),
                callback=callback,
                user_data=name,
            )

        elif item["type"] == "int":
            dpg.add_slider_int(
                label=name,
                default_value=default,
                min_value=item.get("min", 0),
                max_value=item.get("max", 100),
                callback=callback,
                user_data=name,
            )

        elif item["type"] == "bool":
            dpg.add_checkbox(
                label=name,
                default_value=default,
                callback=callback,
                user_data=name,
            )

        elif item["type"] == "button":
            def button_cb():
                print(f"[BUTTON PRESSED] {name}")

            dpg.add_button(label=name, callback=button_cb)







#----------------------------
# Initialize Robot Object
#----------------------------
my_a     = [0.0, 7.375, 0.0,  0.0,  0.0,  0.0]
my_alpha = [np.deg2rad(90),  np.deg2rad(180), np.deg2rad(90),
            np.deg2rad(90),  np.deg2rad(-90), np.deg2rad(0)]
my_d     = [-3.5, 0.0, 0.0, 8.25, 0.0, 5.1875]
my_theta = [np.deg2rad(0),   np.deg2rad(0),   np.deg2rad(90),
            np.deg2rad(180), np.deg2rad(0),   np.deg2rad(-90)]
my_bounds = [
    (np.deg2rad(-90),  np.deg2rad(90)),
    (np.deg2rad(-180), np.deg2rad(0)),
    (np.deg2rad(-90),  np.deg2rad(90)),
    (np.deg2rad(-90),  np.deg2rad(90)),
    (np.deg2rad(-90),  np.deg2rad(90)),
    (np.deg2rad(-90),  np.deg2rad(90)),
]
robot = Robot_math(a=my_a, alpha=my_alpha, d=my_d, theta=my_theta,
                    joint_type=["r"]*6, bounds=my_bounds,
                    fail_dist=[0.1]*6)

robot.WT = robot.make_homogenous_transformation(
    yaw=0, pitch=0, roll=180, x=0, y=0, z=0)            

#----------------------------
# Initialize Solver Object
#----------------------------
solver = Oracle(robot_class= robot)

#----------------------------
# Initialize Path Planner Object
#----------------------------
path_planner = PathPlannerMath(my_robot= robot)

#----------------------------
# Initialize Visualizer Object
#----------------------------
viz = Robot_Visualizer(robot= robot, my_oracle= solver)

# ---------------------------------------
# Control Pannel Configuration (Grouped)
# ---------------------------------------
config = [
    {
        "group": "Loss Weights",
        "items": [
            {"name": "position_w", "type": "float", "default": 1.0, "min": 0, "max": 100},
            {"name": "orientation_w", "type": "float", "default": 0.5, "min": 0, "max": 10},
            {"name": "closeset_dist_w", "type": "float", "default": 0.5, "min": 0, "max": 10},
            {"name": "collision_w", "type": "float", "default": 2.0, "min": 0, "max": 20},

            {"name": "joint_vel_w1", "type": "float", "default": 0.1, "min": 0, "max": 10},
            {"name": "joint_vel_w2", "type": "float", "default": 0.1, "min": 0, "max": 10},
            {"name": "joint_vel_w3", "type": "float", "default": 0.1, "min": 0, "max": 10},
            {"name": "joint_vel_w4", "type": "float", "default": 0.1, "min": 0, "max": 10},
            {"name": "joint_vel_w5", "type": "float", "default": 0.1, "min": 0, "max": 10},
            {"name": "joint_vel_w6", "type": "float", "default": 0.1, "min": 0, "max": 10},

            {"name": "joint_acc_w1", "type": "float", "default": 0.1, "min": 0, "max": 10},
            {"name": "joint_acc_w2", "type": "float", "default": 0.1, "min": 0, "max": 10},
            {"name": "joint_acc_w3", "type": "float", "default": 0.1, "min": 0, "max": 10},
            {"name": "joint_acc_w4", "type": "float", "default": 0.1, "min": 0, "max": 10},
            {"name": "joint_acc_w5", "type": "float", "default": 0.1, "min": 0, "max": 10},
            {"name": "joint_acc_w6", "type": "float", "default": 0.1, "min": 0, "max": 10},
        ],
    },

    {
        "group": "Pass Criteria",
        "items": [
            {"name": "max_good_dist", "type": "float", "default": 0.25, "min": 0, "max": 1},
            {"name": "max_good_ori_deg", "type": "float", "default": 5, "min": 0, "max": 15},

            {"name": "max_okay_dist", "type": "float", "default": 0.5, "min": 0, "max": 1.5},
            {"name": "max_okay_ori_deg", "type": "float", "default": 15, "min": 0, "max": 30},
        ],
    },

    {
        "group": "Controls",
        "items": [
            {"name": "IK_target_X", "type": "float", "default": 0, "min": -robot.max_reach, "max": robot.max_reach},
            {"name": "IK_target_Y", "type": "float", "default": 0, "min": -robot.max_reach, "max": robot.max_reach},
            {"name": "IK_target_Z", "type": "float", "default": 0, "min": -robot.max_reach, "max": robot.max_reach},
            {"name": "IK_target_ROLL", "type": "float", "default": 0, "min": -90, "max": 90},
            {"name": "IK_target_PITCH", "type": "float", "default": 0, "min": -90, "max": 90},
            {"name": "IK_target_YAW", "type": "float", "default": 0, "min": -90, "max": 90},

            {"name": "solve_IK", "type": "button"},

            {"name": "randomize_Start_path_planning", "type": "button"},
            {"name": "randomize_End_path_planning", "type": "button"},

            {"name": "solve_path_planning", "type": "button"},
            
            {"name": "Play_IK", "type": "button"},
            {"name": "Play_Trajectory", "type": "button"},
        ],
    },

    {
        "group": "Robot Joints",
        "items": [
            {"name": "j1", "type": "float", "default": 0.0, "min": robot.low_bounds[0], "max": robot.high_bounds[0]},
            {"name": "j2", "type": "float", "default": 0.0, "min": robot.low_bounds[1], "max": robot.high_bounds[1]},
            {"name": "j3", "type": "float", "default": 0.0, "min": robot.low_bounds[2], "max": robot.high_bounds[2]},
            {"name": "j4", "type": "float", "default": 0.0, "min": robot.low_bounds[3], "max": robot.high_bounds[3]},
            {"name": "j5", "type": "float", "default": 0.0, "min": robot.low_bounds[4], "max": robot.high_bounds[4]},
            {"name": "j6", "type": "float", "default": 0.0, "min": robot.low_bounds[5], "max": robot.high_bounds[5]},
        ],
    },
]

# ----------------------------
# Loss Helper
# ----------------------------
def loss_helper(state, key):
    if key == "position_w":
        solver.L.pos_w = state[key]
    if key == "orientation_w":
        solver.L.rot_w = state[key]
    if key == "closeset_dist_w":
        solver.L.dist_w = state[key]
    if key == "collision_w":
        solver.L.crash_w = state[key]

    if key == "joint_vel_w1":
        solver.L.vel_lambda[0] = state[key]
    if key == "joint_vel_w2":
        solver.L.vel_lambda[1] = state[key]
    if key == "joint_vel_w3":
        solver.L.vel_lambda[2] = state[key]
    if key == "joint_vel_w4":
        solver.L.vel_lambda[3] = state[key]
    if key == "joint_vel_w5":
        solver.L.vel_lambda[4] = state[key]
    if key == "joint_vel_w6":
        solver.L.vel_lambda[5] = state[key]

    if key == "joint_acc_w1":
        solver.L.acc_lambda[0] = state[key]
    if key == "joint_acc_w2":
        solver.L.acc_lambda[1] = state[key]
    if key == "joint_acc_w3":
        solver.L.acc_lambda[2] = state[key]
    if key == "joint_acc_w4":
        solver.L.acc_lambda[3] = state[key]
    if key == "joint_acc_w5":
        solver.L.acc_lambda[4] = state[key]
    if key == "joint_acc_w6":
        solver.L.acc_lambda[5] = state[key]

    if key == "max_good_dist":
        solver.L.max_good_err_pos = state[key]
    if key == "max_good_ori_deg":
        solver.L.max_good_err_deg = state[key]
    if key == "max_okay_dist":
        solver.L.max_ok_err_pos = state[key]
    if key == "max_okay_ori_deg":
        solver.L.max_ok_err_deg = state[key]



# ----------------------------
# Robot Helper
# ----------------------------
def robot_helper(state, key):
    if key == "j1":
        robot.q_vect[0] = state[key]
    if key == "j2":
        robot.q_vect[1] = state[key]
    if key == "j3":
        robot.q_vect[2] = state[key]
    if key == "j4":
        robot.q_vect[3] = state[key]
    if key == "j5":
        robot.q_vect[4] = state[key]
    if key == "j6":
        robot.q_vect[5] = state[key]


# ----------------------------
# Path Planner Helper
# ----------------------------
def path_planner_helper(state, key):
    if key == "randomize_Start_path_planning":
        if state[key] == True:
            path_planner.make_random_q_start()
            state[key] = False

    if key == "randomize_End_path_planning":
        if state[key] == True:
            path_planner.make_random_q_end()
            state[key] = False

    if key == "solve_path_planning":
        if state[key] == True:
            solver.follow_trajectory(path_planner.MoveL_mutable_start_stop())
            state[key] = False

    


# ----------------------------
# Solver Helper
# ----------------------------
def solver_helper(state, key):
    if key ==  "IK_target_X":
        solver.current_XYZ_targ[0] = state[key]
    if key ==  "IK_target_Y":
        solver.current_XYZ_targ[1] = state[key]
    if key ==  "IK_target_Z":
        solver.current_XYZ_targ[2] = state[key]

    if key == "IK_target_YAW":
        solver.current_YPR_targ[0] = state[key]
    if key == "IK_target_PITCH":
        solver.current_YPR_targ[1] = state[key]
    if key == "IK_target_ROLL":
        solver.current_YPR_targ[2] = state[key]

    if key == "solve_IK":
        if state[key] == True:
            solver.get_IK_given_targ()
            state[key] = False


# ----------------------------
# Visualizer Helper
# ----------------------------


# ----------------------------
# Worker Loop
# ----------------------------
def worker(state):
    for key in state:
        loss_helper(state= state, key= key)
        robot_helper(state= state, key= key)
        solver_helper(state= state, key= key)
        path_planner_helper(state= state, key= key)




# ----------------------------
# Main
# ----------------------------
def main():
    state = SharedState()

    dpg.create_context()

    ui = LiveUI(state, config)
    ui.build()

    dpg.create_viewport(title="IK Tuning UI", width=420, height=640)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # threading.Thread(target=print_loop, args=(state,), daemon=True).start()

    dpg.start_dearpygui()
    dpg.destroy_context()

    # for key in state:



if __name__ == "__main__":
    main()