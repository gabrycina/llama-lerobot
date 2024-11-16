import numpy as np

class SimRobotController:
    def __init__(self, env):
        self.env = env
        self.num_joints = 6  # MOSS arm joints
        
        # Define step size for each joint (in radians)
        self.delta = 0.2  # Small increment in radians
        
        # Initialize environment
        obs, _ = self.env.reset()
        self.current_positions = obs['arm_qpos'].copy()
        
        # Initialize action vector
        self.action = np.zeros(6)
        
    def move_joint(self, joint_idx, direction):
        """
        Move a specific joint up or down by delta amount
        joint_idx: 0-5 for the 6 joints
        direction: 1 for up, -1 for down
        """
        if not 0 <= joint_idx < self.num_joints:
            return
            
        # print(f"Old position {self.current_positions}")
            
        # Update the action for the specified joint
        self.action[joint_idx] += direction * self.delta
        
        # Apply the action to the environment
        obs, reward, terminated, truncated, info = self.env.step(self.action)
        
        # print(f"Actual new position {obs['arm_qpos']}")
        # print(f"Delta achieved: {obs['arm_qpos'] - self.current_positions}")
        
        self.current_positions = obs['arm_qpos'].copy()
        
        return obs, reward, terminated, truncated, info

    def get_state(self):
        """Get current state of the robot"""
        return {
            'joint_positions': self.current_positions,
        }

