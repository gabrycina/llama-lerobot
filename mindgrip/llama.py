import os
import time
import base64
import cv2
import time
import json

from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

load_dotenv("../.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


BASIC_PROMPT = """
    You are the robot controller:
    You have 4 functions to move the robot 'up', 'down', 'rotate_left', 'rotate_right', these functions move the end effector in the specified direction.
    I want you to move the robot to get closer to the red cube, only closer, do not perform any other action.
    Strategies you should use:
    - if the end effector is higher than the cube use "down"
    - if the cube is to the left of the end effector use "rotate_left", rotate is always referring to the point of view of the robot
    - if the cube is to the right of the end effector use "rotate_right", rotate is always referring to the point of view of the robot
    If you see that your actions are moving you farder from the cube, you should reverse the action.
    Answer in json format containig the name of the funtion: {"action": "rotate_left"} or {"action": "down"}
"""

DESCRIPTION_PROMPT = """
    Describe the image from a spatial point of view, where is located the robot relative to the mug in the 3 space coordinates?
"""

BASIC_DESCRIPTION = "Describe what you see"

class LlamaPolicy:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.camera = self.init_camera(0)

    def init_camera(self, device_id: int = 0):
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return None
        time.sleep(3)
        return cap

    def capture_image(self, camera, output_dir: str = "assets/capture", device_id: int = 0):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ret, frame = camera.read()
        if not ret:
            camera.release()
            raise Exception("Could not capture image.")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = str(Path(output_dir) / filename)

        cv2.imwrite(filepath, frame)
        print(f"Image saved to: {filepath}")
        return filepath
    
    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def chat_completion(self):
        image_path = self.capture_image(self.camera)
        image = self.encode_image(image_path)
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text":BASIC_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
                    ],
                }
            ],
            response_format= {"type": "json_object"},
            model="llama-3.2-90b-vision-preview",
            temperature=0.0
        )

        result = json.loads(chat_completion.choices[0].message.content)
        print(result)
        return result 

def list_available_cameras(max_devices: int = 10) -> list[int]:
    available_devices = []
    for device_id in range(max_devices):
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            available_devices.append(device_id)
            cap.release()
    return available_devices
        
        