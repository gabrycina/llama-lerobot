import torch
import time
import streamlit as st
import threading
import subprocess
from transformers import pipeline
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from mindgrip.llama import LlamaPolicy

leader_arms = {
        "main": FeetechMotorsBus(
            port="/dev/tty.usbmodem58760430811",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "sts3215"),
                "shoulder_lift": (2, "sts3215"), 
                "elbow_flex": (3, "sts3215"),
                "wrist_flex": (4, "sts3215"),
                "wrist_roll": (5, "sts3215"),
                "gripper": (6, "sts3215"),
            },
        ),
    }

follower_arms = {
        "main": FeetechMotorsBus(
            port="/dev/tty.usbmodem58760432661",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "sts3215"),
                "shoulder_lift": (2, "sts3215"),
                "elbow_flex": (3, "sts3215"), 
                "wrist_flex": (4, "sts3215"),
                "wrist_roll": (5, "sts3215"),
                "gripper": (6, "sts3215"),
            },
        ),
    }

def run_streamlit():
    """Run the Streamlit app in a separate process"""
    subprocess.Popen(["streamlit", "run", "mindgrip/streamlit_app.py"])

class RealRobotController:
    def __init__(self):
        self.robot = ManipulatorRobot(
            robot_type="moss",
            calibration_dir="../.cache/calibration/moss",
            leader_arms=leader_arms,
            follower_arms=follower_arms,
        )
        
        self.robot.connect()
        
        print("Moving to zero position...")
        # Initialize to zero position with exact calibration values
        zero_position = {
            'shoulder_pan': 30,    # From calibration
            'shoulder_lift': 90,    # From calibration
            'elbow_flex': 70,      # From calibration
            'wrist_flex': 20,      # From calibration
            'wrist_roll': 10,      # From calibration
            'gripper': 0         # From calibration
        }
        
        # Move to zero position one joint at a time
        for motor_name, position in zero_position.items():
            self.robot.follower_arms['main'].write("Goal_Position", [position], [motor_name])
            time.sleep(1.0)  # Longer delay for safer movement
        
        # Define very gentle relative movements
        self.relative_actions = {
            'up': {
                'shoulder_lift': -10,    # Very gentle lift
                'elbow_flex': -15,       # Small elbow compensation
                'wrist_flex': 10,        # Keep end effector level
                'wrist_roll': -5         # Slight roll compensation
            },
            'down': {
                'shoulder_lift': 10,     # Very gentle lower
                'elbow_flex': 15,        # Small elbow compensation
                'wrist_flex': -10,       # Keep end effector level
                'wrist_roll': 5          # Slight roll compensation
            },
            'rotate_left': {
                'shoulder_pan': 15       # Very gentle rotate
            },
            'rotate_right': {
                'shoulder_pan': -15      # Very gentle rotate
            }
        }
        
        # Slower movement speed for more control
        self.movement_speed = 50  # Range: 0-1023
        
        # Add reference to LlamaPolicy for camera access
        self.llama = LlamaPolicy(camera_id=2)
        
        # Start Streamlit
        run_streamlit()

    def execute_relative_movement(self, action_name):
        """Execute a relative movement from current position"""
        if action_name not in self.relative_actions:
            print(f"Unknown action: {action_name}")
            return
            
        # Get current positions
        current_positions = {}
        for name in self.robot.follower_arms:
            current_positions = self.robot.follower_arms[name].read("Present_Position")
            
        # Calculate new positions
        new_positions = current_positions.copy()
        for motor, delta in self.relative_actions[action_name].items():
            motor_idx = list(self.robot.follower_arms['main'].motors.keys()).index(motor)
            new_positions[motor_idx] += delta
            
            # Ensure within safe limits (0 to 4096)
            new_positions[motor_idx] = max(0, min(4096, new_positions[motor_idx]))
        
        # Execute movement
        for name in self.robot.follower_arms:
            self.robot.follower_arms[name].write("Goal_Position", new_positions)
            
    def handle_command(self, text):
        """Handle movement commands"""
        # Direct mapping of commands to actions
        command_mapping = {
            'up': 'up',
            'down': 'down',
            'rotate_left': 'rotate_left',
            'rotate_right': 'rotate_right',
            'forward': 'forward',
            'backward': 'backward'
        }
        
        text = text.lower()
        for command, action in command_mapping.items():
            if command in text:
                print(f"Executing {action} movement")
                self.execute_relative_movement(action)
                return
                
        print("Command not recognized")

def main():
    controller = RealRobotController()
    
    print("\nReady to accept commands!")
    print("Available commands: up, down, left, right, forward, backward")
    print("Camera feed available at http://localhost:8501")
    
    for i in range(8):
        result = controller.llama.chat_completion()
        
        if result["action"] in ["up", "down", "rotate_left", "rotate_right"]:
            controller.handle_command(result["action"])
        else:
            print(f"Error {result} is not mapped ")
        
        time.sleep(1)

if __name__ == "__main__":
    main()