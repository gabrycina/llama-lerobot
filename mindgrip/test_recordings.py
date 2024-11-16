import torch
import time
from transformers import pipeline
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

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

class RealRobotController:
    def __init__(self):
        self.robot = ManipulatorRobot(
            robot_type="moss",
            calibration_dir="../.cache/calibration/moss",
            leader_arms=leader_arms,
            follower_arms=follower_arms,
        )
        
        self.robot.connect()
        
        # Define smaller relative movements for incremental control
        # Note: 50 steps ≈ 4.4 degrees (4096 steps = 360 degrees)
        self.relative_actions = {
            'up': {
                'shoulder_lift': -50,   # Small lift
                'elbow_flex': -50,      # Small elbow compensation
                'wrist_flex': 50        # Keep end effector level
            },
            'down': {
                'shoulder_lift': 50,     # Small lower
                'elbow_flex': 50,        # Small elbow compensation
                'wrist_flex': -50        # Keep end effector level
            },
            'left': {
                'shoulder_pan': 50       # Small rotate left
            },
            'right': {
                'shoulder_pan': -50      # Small rotate right
            },
            'forward': {
                'shoulder_lift': -25,    # Tiny shoulder lift
                'elbow_flex': -75,       # Small extend
                'wrist_flex': 25         # Keep end effector level
            },
            'backward': {
                'shoulder_lift': 25,     # Tiny shoulder drop
                'elbow_flex': 75,        # Small retract
                'wrist_flex': -25        # Keep end effector level
            }
        }
        
        # Slower movement speed for more control
        self.movement_speed = 50  # Range: 0-1023
        
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
            'left': 'left',
            'right': 'right',
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
    
    while True:
        command = input("\nEnter command (or 'quit' to exit): ")
        if command.lower() == 'quit':
            break
        controller.handle_command(command)

if __name__ == "__main__":
    main()
    
    
# def control(robot: Robot | None = None, **kwargs):
#     print("Initializing robot...")
#     if robot is None:
#         raise ValueError("Robot must be provided")
    
#     controller = RealRobotController(robot)
#     print("\nControl Commands:")
#     print("1-6: Select joint")
#     print("+ : Move selected joint up")
#     print("- : Move selected joint down")
#     print("s : Stop current movement")
#     print("q : Quit")
    
#     selected_joint = 0  # Default to first joint
#     last_command = None
    
#     import sys
#     import select
#     import tty
#     import termios

#     # Set up terminal for non-blocking input
#     old_settings = termios.tcgetattr(sys.stdin)
#     tty.setcbreak(sys.stdin.fileno())
    
#     try:
#         while True:
#             # Check if there's input available
#             if select.select([sys.stdin], [], [], 0.0)[0]:
#                 key = sys.stdin.read(1)
                
#                 if key == 'q':
#                     break
#                 elif key in ['1', '2', '3', '4', '5', '6']:
#                     selected_joint = int(key) - 1
#                     last_command = None  # Stop current movement when changing joint
#                     print(f"\nSelected joint {selected_joint + 1}")
#                 elif key == '+':
#                     last_command = (selected_joint, 50)  # 50 steps ≈ 4.4 degrees
#                 elif key == '-':
#                     last_command = (selected_joint, -50)  # -50 steps ≈ -4.4 degrees
#                 elif key == 's':
#                     last_command = None
#                     print("\nStopped movement")
            
#             # Execute last command if there is one
#             if last_command is not None:
#                 joint, steps = last_command
#                 # Get current positions
#                 current_positions = robot.follower_arms["main"].read("Present_Position")
                
#                 # Update position for selected joint
#                 new_positions = current_positions.copy()
#                 new_positions[joint] += steps
                
#                 # Ensure within safe limits (0 to 4096)
#                 new_positions[joint] = max(0, min(4096, new_positions[joint]))
                
#                 # Execute movement
#                 robot.follower_arms["main"].write("Goal_Position", new_positions)
            
#             # Display current state
#             positions = robot.follower_arms["main"].read("Present_Position")
#             print(f"\rCurrent Joint Positions: {positions}, Selected Joint: {selected_joint + 1}", end="")
            
#     finally:
#         # Restore terminal settings
#         termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
#         robot.disconnect()

# class RealRobotController:
#     def __init__(self, robot):
#         self.robot = robot