#!/home/pushpak/anaconda3/envs/cmu/bin/python

import os
import math
import copy
import json
import actionlib
import control_msgs.msg
from controller import ArmController
from gazebo_msgs.msg import ModelStates
import rospy
from pyquaternion import Quaternion as PyQuaternion
import numpy as np
from gazebo_ros_link_attacher.srv import SetStatic, SetStaticRequest, SetStaticResponse
from gazebo_ros_link_attacher.srv import Attach, AttachRequest, AttachResponse
from openai import OpenAI
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

PKG_PATH = os.path.dirname(os.path.abspath(__file__))

# picked_lego_name = None

MODELS_INFO = {
    "X1-Y2-Z1": {
        "home": [0.264589, -0.293903, 0.777] 
    },
    "X2-Y2-Z2": {
        "home": [0.277866, -0.724482, 0.777] 
    },
    "X1-Y3-Z2": {
        "home": [0.268053, -0.513924, 0.777]  
    },
    "X1-Y2-Z2": {
        "home": [0.429198, -0.293903, 0.777] 
    },
    "X1-Y2-Z2-CHAMFER": {
        "home": [0.592619, -0.293903, 0.777]  
    },
    "X1-Y4-Z2": {
        "home": [0.108812, -0.716057, 0.777] 
    },
    "X1-Y1-Z2": {
        "home": [0.088808, -0.295820, 0.777] 
    },
    "X1-Y2-Z2-TWINFILLET": {
        "home": [0.103547, -0.501132, 0.777] 
    },
    "X1-Y3-Z2-FILLET": {
        "home": [0.433739, -0.507130, 0.777]  
    },
    "X1-Y4-Z1": {
        "home": [0.589908, -0.501033, 0.777]  
    },
    "X2-Y2-Z2-FILLET": {
        "home": [0.442505, -0.727271, 0.777] 
    }
}

for model, model_info in MODELS_INFO.items():
    pass
    #MODELS_INFO[model]["home"] = model_info["home"] + np.array([0.0, 0.10, 0.0])

for model, info in MODELS_INFO.items():
    model_json_path = os.path.join(PKG_PATH, "..", "models", f"lego_{model}", "model.json")
    # make path absolute
    model_json_path = os.path.abspath(model_json_path)
    # check path exists
    if not os.path.exists(model_json_path):
        raise FileNotFoundError(f"Model file {model_json_path} not found")

    model_json = json.load(open(model_json_path, "r"))
    corners = np.array(model_json["corners"])

    size_x = (np.max(corners[:, 0]) - np.min(corners[:, 0]))
    size_y = (np.max(corners[:, 1]) - np.min(corners[:, 1]))
    size_z = (np.max(corners[:, 2]) - np.min(corners[:, 2]))

    #print(f"{model}: {size_x:.3f} x {size_y:.3f} x {size_z:.3f}")

    MODELS_INFO[model]["size"] = (size_x, size_y, size_z)

# Compensate for the interlocking height
INTERLOCKING_OFFSET = 0.019

SAFE_X = -0.40
SAFE_Y = -0.13
SURFACE_Z = 0.774  # Set this to a known safe value for your environment

# Resting orientation of the end effector
DEFAULT_QUAT = PyQuaternion(axis=(0, 1, 0), angle=math.pi)
# Resting position of the end effector
DEFAULT_POS = (-0.1, -0.2, 1.2)

DEFAULT_PATH_TOLERANCE = control_msgs.msg.JointTolerance()
DEFAULT_PATH_TOLERANCE.name = "path_tolerance"
DEFAULT_PATH_TOLERANCE.velocity = 10
DEFAULT_QUAT = PyQuaternion(axis=(0, 1, 0), angle=math.pi)

# Set up OpenAI API
# openai.api_key = os.getenv("OPENAI_API_KEY")

def get_gazebo_model_name(model_name, vision_model_pose):
    """
        Get the name of the model inside gazebo. It is needed for link attacher plugin.
    """
    models = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=None)
    epsilon = 0.05
    for gazebo_model_name, model_pose in zip(models.name, models.pose):
        if model_name not in gazebo_model_name:
            continue
        # Get everything inside a square of side epsilon centered in vision_model_pose
        ds = abs(model_pose.position.x - vision_model_pose.position.x) + abs(model_pose.position.y - vision_model_pose.position.y)
        if ds <= epsilon:
            return gazebo_model_name
    raise ValueError(f"Model {model_name} at position {vision_model_pose.position.x} {vision_model_pose.position.y} was not found!")


def get_model_name(gazebo_model_name):
    return gazebo_model_name.replace("lego_", "").split("_", maxsplit=1)[0]


def get_legos_pos(vision=False):
    if vision:
        try:
            print("Attempting to get lego detections from vision...")
            legos = rospy.wait_for_message("/lego_detections", ModelStates, timeout=5.0)
            print(f"Received lego detections: {legos}")
            return [(lego_name, lego_pose) for lego_name, lego_pose in zip(legos.name, legos.pose)]
        except rospy.ROSException as e:
            print(f"Timeout waiting for lego detections: {e}")
            return []
    else:
        print("Getting model states from Gazebo...")
        models = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=None)
        legos = ModelStates()

        print("All Gazebo models:")
        for name, pose in zip(models.name, models.pose):
            print(f"  {name}: {pose}")
            if "X" in name:
                name = get_model_name(name)
                legos.name.append(name)
                legos.pose.append(pose)
        
        print(f"Filtered Gazebo model states: {legos}")
        return [(lego_name, lego_pose) for lego_name, lego_pose in zip(legos.name, legos.pose)]


def straighten(model_pose, gazebo_model_name):
    x = model_pose.position.x
    y = model_pose.position.y
    z = model_pose.position.z
    model_quat = PyQuaternion(
        x=model_pose.orientation.x,
        y=model_pose.orientation.y,
        z=model_pose.orientation.z,
        w=model_pose.orientation.w)

    model_size = MODELS_INFO[get_model_name(gazebo_model_name)]["size"]

    """
        Calculate approach quaternion and target quaternion
    """

    facing_direction = get_axis_facing_camera(model_quat)
    approach_angle = get_approach_angle(model_quat, facing_direction)

    print(f"Lego is facing {facing_direction}")
    print(f"Angle of approaching measures {approach_angle:.2f} deg")

    # Calculate approach quat
    approach_quat = get_approach_quat(facing_direction, approach_angle)

    # Get above the object
    controller.move_to(x, y, target_quat=approach_quat)

    # Calculate target quat
    regrip_quat = DEFAULT_QUAT
    if facing_direction == (1, 0, 0) or facing_direction == (0, 1, 0):  # Side
        target_quat = DEFAULT_QUAT
        pitch_angle = -math.pi/2 + 0.2

        if abs(approach_angle) < math.pi/2:
            target_quat = target_quat * PyQuaternion(axis=(0, 0, 1), angle=math.pi/2)
        else:
            target_quat = target_quat * PyQuaternion(axis=(0, 0, 1), angle=-math.pi/2)
        target_quat = PyQuaternion(axis=(0, 1, 0), angle=pitch_angle) * target_quat

        if facing_direction == (0, 1, 0):
            regrip_quat = PyQuaternion(axis=(0, 0, 1), angle=math.pi/2) * regrip_quat

    elif facing_direction == (0, 0, -1):
        """
            Pre-positioning
        """
        controller.move_to(z=z, target_quat=approach_quat)
        close_gripper(gazebo_model_name, model_size[0])

        tmp_quat = PyQuaternion(axis=(0, 0, 1), angle=2*math.pi/6) * DEFAULT_QUAT
        controller.move_to(SAFE_X, SAFE_Y, z+0.05, target_quat=tmp_quat, z_raise=0.1)  # Move to safe position
        controller.move_to(z=z)
        open_gripper(gazebo_model_name)

        approach_quat = tmp_quat * PyQuaternion(axis=(1, 0, 0), angle=math.pi/2)

        target_quat = approach_quat * PyQuaternion(axis=(0, 0, 1), angle=-math.pi)  # Add a yaw rotation of 180 deg

        regrip_quat = tmp_quat * PyQuaternion(axis=(0, 0, 1), angle=math.pi)
    else:
        target_quat = DEFAULT_QUAT
        target_quat = target_quat * PyQuaternion(axis=(0, 0, 1), angle=-math.pi/2)

    """
        Grip the model
    """
    if facing_direction == (0, 0, 1) or facing_direction == (0, 0, -1):
        closure = model_size[0]
        z = SURFACE_Z + model_size[2] / 2
    elif facing_direction == (1, 0, 0):
        closure = model_size[1]
        z = SURFACE_Z + model_size[0] / 2
    elif facing_direction == (0, 1, 0):
        closure = model_size[0]
        z = SURFACE_Z + model_size[1] / 2
    controller.move_to(z=z, target_quat=approach_quat)
    close_gripper(gazebo_model_name, closure)

    """
        Straighten model if needed
    """
    if facing_direction != (0, 0, 1):
        z = SURFACE_Z + model_size[2]/2

        controller.move_to(z=z+0.05, target_quat=target_quat, z_raise=0.1)
        controller.move(dz=-0.05)
        open_gripper(gazebo_model_name)

        # Re grip the model
        controller.move_to(z=z, target_quat=regrip_quat, z_raise=0.1)
        close_gripper(gazebo_model_name, model_size[0])


def close_gripper(gazebo_model_name, closure=0):
    set_gripper(0.81-closure*10)
    rospy.sleep(0.5)
    # Create dynamic joint
    if gazebo_model_name is not None:
        req = AttachRequest()
        req.model_name_1 = gazebo_model_name
        req.link_name_1 = "link"
        req.model_name_2 = "robot"
        req.link_name_2 = "wrist_3_link"
        attach_srv.call(req)


def open_gripper(gazebo_model_name=None):
    print(f"Starting open_gripper function for model: {gazebo_model_name}")
    start_time = time.time()
    
    try:
        set_gripper(0.0)
        print(f"Gripper set to open position. Time taken: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error setting gripper to open position: {e}")
        return

    # Destroy dynamic joint
    if gazebo_model_name is not None:
        print(f"Attempting to detach model: {gazebo_model_name}")
        detach_start_time = time.time()
        req = AttachRequest()
        req.model_name_1 = gazebo_model_name
        req.link_name_1 = "link"
        req.model_name_2 = "robot"
        req.link_name_2 = "wrist_3_link"
        try:
            detach_srv.call(req)
            print(f"Model detached successfully. Time taken: {time.time() - detach_start_time:.2f} seconds")
        except Exception as e:
            print(f"Error detaching model: {e}")
    
    total_time = time.time() - start_time
    print(f"open_gripper function completed. Total time: {total_time:.2f} seconds")


def set_model_fixed(model_name):
    req = AttachRequest()
    req.model_name_1 = model_name
    req.link_name_1 = "link"
    req.model_name_2 = "ground_plane"
    req.link_name_2 = "link"
    attach_srv.call(req)

    req = SetStaticRequest()
    print("{} TO HOME".format(model_name))
    req.model_name = model_name
    req.link_name = "link"
    req.set_static = True

    setstatic_srv.call(req)


def get_approach_quat(facing_direction, approach_angle):
    quat = DEFAULT_QUAT
    if facing_direction == (0, 0, 1):
        pitch_angle = 0
        yaw_angle = 0
    elif facing_direction == (1, 0, 0) or facing_direction == (0, 1, 0):
        pitch_angle = + 0.2
        if abs(approach_angle) < math.pi/2:
            yaw_angle = math.pi/2
        else:
            yaw_angle = -math.pi/2
    elif facing_direction == (0, 0, -1):
        pitch_angle = 0
        yaw_angle = 0
    else:
        raise ValueError(f"Invalid model state {facing_direction}")

    quat = quat * PyQuaternion(axis=(0, 1, 0), angle=pitch_angle)
    quat = quat * PyQuaternion(axis=(0, 0, 1), angle=yaw_angle)
    quat = PyQuaternion(axis=(0, 0, 1), angle=approach_angle+math.pi/2) * quat

    return quat


def get_axis_facing_camera(quat):
    axis_x = np.array([1, 0, 0])
    axis_y = np.array([0, 1, 0])
    axis_z = np.array([0, 0, 1])
    new_axis_x = quat.rotate(axis_x)
    new_axis_y = quat.rotate(axis_y)
    new_axis_z = quat.rotate(axis_z)
    # get angle between new_axis and axis_z
    angle = np.arccos(np.clip(np.dot(new_axis_z, axis_z), -1.0, 1.0))
    # get if model is facing up, down or sideways
    if angle < np.pi / 3:
        return 0, 0, 1
    elif angle < np.pi / 3 * 2 * 1.2:
        if abs(new_axis_x[2]) > abs(new_axis_y[2]):
            return 1, 0, 0
        else:
            return 0, 1, 0
        #else:
        #    raise Exception(f"Invalid axis {new_axis_x}")
    else:
        return 0, 0, -1


def get_approach_angle(model_quat, facing_direction):#get gripper approach angle
    if facing_direction == (0, 0, 1):
        return model_quat.yaw_pitch_roll[0] - math.pi/2 #rotate gripper
    elif facing_direction == (1, 0, 0) or facing_direction == (0, 1, 0):
        axis_x = np.array([0, 1, 0])
        axis_y = np.array([-1, 0, 0])
        new_axis_z = model_quat.rotate(np.array([0, 0, 1])) #get z axis of lego
        # get angle between new_axis and axis_x
        dot = np.clip(np.dot(new_axis_z, axis_x), -1.0, 1.0) #sin angle between lego z axis and x axis in fixed frame
        det = np.clip(np.dot(new_axis_z, axis_y), -1.0, 1.0) #cos angle between lego z axis and x axis in fixed frame
        return math.atan2(det, dot) #get angle between lego z axis and x axis in fixed frame
    elif facing_direction == (0, 0, -1):
        return -(model_quat.yaw_pitch_roll[0] - math.pi/2) % math.pi - math.pi
    else:
        raise ValueError(f"Invalid model state {facing_direction}")


def set_gripper(value):
    goal = control_msgs.msg.GripperCommandGoal()
    goal.command.position = value  # From 0.0 to 0.8
    goal.command.max_effort = -1  # # Do not limit the effort
    action_gripper.send_goal_and_wait(goal, rospy.Duration(10))

    return action_gripper.get_result()


def process_command(command):
    print(f"Received command: {command}")
    try:
        print("Sending request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI assistant controlling a UR5 robot with a Robotiq gripper. The robot can manipulate Lego blocks of different colors and sizes. Respond with a JSON array of steps to execute the given command. Each step should be an object with 'action', 'color', and 'location' fields. Valid actions are 'scan', 'pick', 'place', and 'return_home'."},
                {"role": "user", "content": command}
            ]
        )
        print("Received response from OpenAI API")
        content = response.choices[0].message.content
        print(f"Response content: {content}")

        # Strip out code block markers if present
        if content.startswith("```") and content.endswith("```"):
            content = content.strip("```").strip()
        if content.startswith("json"):
            content = content[4:].strip()

        print("Parsing JSON response...")
        steps = json.loads(content)
        print(f"Parsed steps: {steps}")

        for step in steps:
            action = step.get('action')
            color = step.get('color', None)
            location = step.get('location', None)
            print(f"Executing step: action={action}, color={color}, location={location}")

            try:
                if action == 'scan':
                    scan_environment()
                elif action == 'pick':
                    success = pick_lego(color)
                    if not success:
                        print("Pick action failed. Stopping command execution.")
                        break
                elif action == 'place':
                    place_lego(color, location)
                elif action == 'return_home':
                    return_to_home()
                else:
                    print(f"Unrecognized action: {action}")
            except Exception as e:
                print(f"Error executing action {action}: {e}")
                break

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("Raw response content:", content)
    except Exception as e:
        print(f"An error occurred: {e}")
    print("Command processing completed")

picked_lego_name = None

def pick_lego(color):
    global picked_lego_name, picked_lego_pose
    legos = scan_environment()
    if not legos:
        print("No Legos found in the environment.")
        return False

    print(f"Searching for a {color} Lego...")
    target_lego = next((lego for lego in legos if (color is None or color.lower() in ['any', 'unspecified', 'specific'] or color.upper() in lego[0])), None)
    
    if target_lego:
        model_name, model_pose = target_lego
        print(f"Found target Lego: {model_name} at position {model_pose.position}")
        gazebo_model_name = get_gazebo_model_name(model_name, model_pose)
        straighten(model_pose, gazebo_model_name)
        controller.move(dz=0.15)
        picked_lego_name = gazebo_model_name
        picked_lego_pose = model_pose  # Store the picked Lego's pose
        print(f"Picked up the {model_name} Lego.")
        return True
    else:
        print(f"No {color} Lego found." if color else "No Lego found.")
        print(f"Available Legos: {[lego[0] for lego in legos]}")
        return False

import time
import threading

def open_gripper_async(gazebo_model_name):
    threading.Thread(target=open_gripper, args=(gazebo_model_name,)).start()

def place_lego(color, location):
    global picked_lego_name, picked_lego_pose
    print(f"Starting place_lego function for {color if color else 'unspecified'} Lego")
    
    if picked_lego_pose is None:
        print("Error: No Lego has been picked up.")
        return

    x = picked_lego_pose.position.x
    y = picked_lego_pose.position.y
    z = SURFACE_Z
    z_place = z + 0.01  # 1 cm above the surface
    z_up = z + 0.15  # 15 cm above the surface

    print(f"Moving to place Lego: x={x}, y={y}, z={z_place}")
    controller.move_to(x, y, z_place + 0.05, target_quat=DEFAULT_QUAT)  # Stop 5cm above placement

    print("Initiating gripper opening and final approach")
    gripper_start_time = time.time()
    open_gripper_async(picked_lego_name)  # Start opening gripper asynchronously
    
    controller.move_to(x, y, z_place, target_quat=DEFAULT_QUAT)  # Complete final approach
    
    print("Initiating upward movement")
    move_up_start_time = time.time()
    controller.move_to(x, y, z_up, target_quat=DEFAULT_QUAT)
    move_up_end_time = time.time()
    
    print(f"Moved up to z={z_up}. Time taken: {move_up_end_time - move_up_start_time:.2f} seconds")
    print(f"Total time from initiating gripper opening to completion of upward movement: {move_up_end_time - gripper_start_time:.2f} seconds")

    print(f"Successfully placed the {color if color else ''} Lego on the table at x={x}, y={y}.")
    picked_lego_name = None
    picked_lego_pose = None
    print("place_lego function completed")

def stack_legos(color, location):
    legos = scan_environment()
    target_legos = [lego for lego in legos if color.upper() in lego[0]]
    
    if len(target_legos) < 2:
        print(f"Not enough {color} Legos to stack.")
        return
    
    if location == 'right':
        x, y = 0.5, -0.3
    elif location == 'left':
        x, y = -0.5, -0.3
    else:  # center
        x, y = 0, -0.3
    
    for i, lego in enumerate(target_legos):
        model_name, model_pose = lego
        gazebo_model_name = get_gazebo_model_name(model_name, model_pose)
        straighten(model_pose, gazebo_model_name)
        controller.move(dz=0.15)
        
        z = SURFACE_Z + (i * 0.05)  # Assuming each Lego is about 5cm tall
        
        controller.move_to(x, y, z, target_quat=DEFAULT_QUAT)
        open_gripper(gazebo_model_name)
        controller.move(dz=0.15)
    
    print(f"Stacked {len(target_legos)} {color} Legos at the {location} location.")

def scan_environment():
    print("Scanning environment...")
    vision_legos = get_legos_pos(vision=True)
    if vision_legos:
        print(f"Detected Legos from vision: {vision_legos}")
        return vision_legos
    else:
        print("No Legos detected by vision system, falling back to Gazebo model states...")
        gazebo_legos = get_legos_pos(vision=False)
        print(f"Detected Legos from Gazebo: {gazebo_legos}")
        return gazebo_legos

def return_to_home():
    controller.move_to(*DEFAULT_POS, DEFAULT_QUAT)
    print("Returned to home position.")

def manual_calibrate_surface_height():
    global SURFACE_Z
    print(f"Current SURFACE_Z: {SURFACE_Z}")
    new_z = input("Enter new SURFACE_Z value (or press Enter to keep current): ")
    if new_z:
        try:
            SURFACE_Z = float(new_z)
            print(f"SURFACE_Z updated to: {SURFACE_Z}")
        except ValueError:
            print("Invalid input. SURFACE_Z not changed.")
    else:
        print("SURFACE_Z not changed.")

if __name__ == "__main__":
    print("Initializing node of kinematics")
    rospy.init_node("send_joints")

    controller = ArmController()

    # Create an action client for the gripper
    action_gripper = actionlib.SimpleActionClient(
        "/gripper_controller/gripper_cmd",
        control_msgs.msg.GripperCommandAction
    )
    print("Waiting for action of gripper controller")
    action_gripper.wait_for_server()

    setstatic_srv = rospy.ServiceProxy("/link_attacher_node/setstatic", SetStatic)
    attach_srv = rospy.ServiceProxy("/link_attacher_node/attach", Attach)
    detach_srv = rospy.ServiceProxy("/link_attacher_node/detach", Attach)
    setstatic_srv.wait_for_service()
    attach_srv.wait_for_service()
    detach_srv.wait_for_service()

    controller.move_to(*DEFAULT_POS, DEFAULT_QUAT)

    print("Available ROS topics:")
    topics = rospy.get_published_topics()
    for topic, topic_type in topics:
        print(f"  {topic}: {topic_type}")

    while not rospy.is_shutdown():
        command = input("Enter a command: ")
        if command.lower() == 'exit':
            break
        elif command.lower() == 'scan':
            scan_environment()
        elif command.lower() == 'return home':
            return_to_home()
        elif command.lower() == 'calibrate':
            manual_calibrate_surface_height()
        else:
            process_command(command)

    print("Exiting program.")