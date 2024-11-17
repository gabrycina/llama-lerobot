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

DESCRIPTION_PROMPT = """
    Describe the image from a spatial point of view, where is located the robot relative to the mug in the 3 space coordinates?
"""

BASIC_DESCRIPTION = "Describe what you see"

class LlamaPolicy:
    def __init__(self, camera_id=0):
        self.client = Groq(api_key="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")
        self.camera = self.init_camera(camera_id)

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
            temperature=0.5
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
        
        