import os
import time
import base64
import cv2
import json
import numpy as np
import pyautogui
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
load_dotenv("../.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

BASIC_PROMPT = """
You are controlling a robotic arm in a simulation environment. Your task is to minimize the Euclidean distance between the robot's end effector and the target is the brown cup. Move the arm closer to it and push it.

Important spatial context:
- You are viewing the simulation from a third-person perspective
- Available actions:
  * "up": Move end effector upward (+Z axis)
  * "down": Move end effector downward (-Z axis)
  * "rotate_left": Rotate counter-clockwise
  * "rotate_right": Rotate clockwise

You must respond JUST with a JSON object containing exactly ONE action that will minimize the Euclidean distance:
{{"action": "up"}} or
{{"action": "down"}} or
{{"action": "rotate_left"}} or
{{"action": "rotate_right"}}
"""

class LlamaPolicy:
    def __init__(self, camera_id=0):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.camera = self.init_camera(camera_id)

    def init_camera(self, device_id: int = 0):
        """Initialize webcam"""
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return None
        time.sleep(3)
        return cap

    def capture_mujoco_window(self, output_dir: str = "assets/capture"):
        """Capture the entire screen on macOS"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Capture the entire screen without specifying region
            screenshot = pyautogui.screenshot()
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"mujoco_window_{timestamp}.jpg"
            filepath = str(Path(output_dir) / filename)
            
            cv2.imwrite(filepath, frame)
            print(f"Screen captured to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error capturing screen: {e}")
            raise

    def capture_webcam(self, camera, output_dir: str = "assets/capture"):
        """Capture from webcam"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ret, frame = camera.read()
        if not ret:
            camera.release()
            raise Exception("Could not capture image.")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"webcam_capture_{timestamp}.jpg"
        filepath = str(Path(output_dir) / filename)

        cv2.imwrite(filepath, frame)
        print(f"Webcam image saved to: {filepath}")
        return filepath

    def capture_image(self, output_dir: str = "assets/capture"):
        """Capture image based on selected mode"""
        if self.mode == "webcam":
            return self.capture_webcam(self.camera, output_dir)
        else:
            return self.capture_mujoco_window(output_dir)
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def chat_completion(self):
        """Get action from vision model"""
        image_path = self.capture_image()
        image = self.encode_image(image_path)
        
        # Add context about previous actions
        context = f"\nPrevious actions taken: {', '.join(self.action_history[-3:]) if self.action_history else 'None'}"
        
        # Combine system message and prompt
        system_message = "You are a robotic control system that analyzes images from a camera behind a robot. Your goal is to guide the robot's end effector to a red cube using only four possible actions: up, down, rotate_left, or rotate_right. Always respond with a single action in JSON format.\n\n"
        full_prompt = system_message + BASIC_PROMPT + context
        
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
                    ],
                }
            ],
            response_format={"type": "json_object"},
            model="llama-3.2-90b-vision-preview",
            temperature=0.5
        )

        result = json.loads(chat_completion.choices[0].message.content)
        if "action" in result:
            self.action_history.append(result["action"])
        print(result)
        return result

    def reset_history(self):
        """Clear action history"""
        self.action_history = []

    def cleanup(self):
        """Clean up resources"""
        if self.camera is not None:
            self.camera.release()

def find_window_position():
    """Helper function to find mouse position for window coordinates"""
    try:
        while True:
            x, y = pyautogui.position()
            print(f'Mouse Position - X: {x} Y: {y}')
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nDone')

def list_available_cameras(max_devices: int = 10) -> list[int]:
    """List all available camera devices"""
    available_devices = []
    for device_id in range(max_devices):
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            available_devices.append(device_id)
            cap.release()
    return available_devices

if __name__ == "__main__":
    # Example usage
    # To find Mujoco window coordinates:
    # find_window_position()
    
    # To use the policy:
    policy = LlamaPolicy(mode="mujoco")  # or mode="webcam"
    try:
        result = policy.chat_completion()
        print(f"Chosen action: {result['action']}")
    finally:
        policy.cleanup()
        
        