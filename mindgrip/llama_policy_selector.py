import sys
import os
import argparse
import subprocess
import time
import base64
import cv2
import io
import glob
from pathlib import Path

# Add cortex-example/python to Python path
cortex_path = os.path.join(os.path.dirname(__file__), 'cortex-example', 'python')
sys.path.append(cortex_path)

# Now we can import LiveAdvance
from live_advance import LiveAdvance

from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config
from lerobot.common.robot_devices.control_utils import init_policy, control_loop
from test_recordings import RealRobotController

import pyttsx3
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
from groq import Groq
import wave

GROQ_API_KEY = "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"

# models = {
#     "grab_sponge":  {"repo_id": "1g0rrr/grab_sponge", "control_time_s": 32},
#      "grab_orange": {"repo_id": "1g0rrr/grab_orange", "control_time_s": 10}, 
#      "grab_candy":{"repo_id": "1g0rrr/grab_candy", "control_time_s": 10}
# }


def run_streamlit():
    """Run the Streamlit app in a separate process"""
    streamlit_path = os.path.join(os.path.dirname(__file__), 'streamlit_app.py')
    subprocess.Popen(["streamlit", "run", streamlit_path])

def do_control_loop(policy_obj, robot, display_cameras = True):
    control_loop(
        robot=robot,
        control_time_s=policy_obj["control_time_s"],
        display_cameras=display_cameras,
        policy=policy_obj["policy"],
        device=policy_obj["device"],
        use_amp=policy_obj["use_amp"],
        fps = policy_obj["policy_fps"],
        teleoperate=False,
    )

def init_robot():
    engine = pyttsx3.init()
    robot_cfg = init_hydra_config("../lerobot/configs/robot/moss.yaml", []) # la lista rappresenta robot_overrides
    robot = make_robot(robot_cfg)
    robot.connect()

    models = {
        "pills-picking": {"repo_id": "fracapuano/moss-pills", "control_time_s": 100},
        "cup-dragging": {"repo_id": "fracapuano/moss-cup", "control_time_s": 100}
    }
    policies = {}
   
    for model_name in models:
        model = models[model_name]
        policy_overrides = ["device=mps"] # dicono loro
        policy, policy_fps, device, use_amp = init_policy(model["repo_id"], policy_overrides)
        policies[model_name] = ({"policy": policy, "policy_fps": policy_fps, "device": device, "use_amp": use_amp, "control_time_s": model["control_time_s"]})

    while True:
        transcript = listen_to_user()
        print(transcript)
        description = describe_the_scene()
        print(f"Description: {description}")
        if description is not None:
            engine.say(description)
            engine.runAndWait()
        policy = select_policy(transcript)
        if policy == "none":
            continue
        
        print(policy)
        do_control_loop(policies[policy], robot)
        input("Next query")

def listen_to_user():
    # Audio recording parameters
    duration = 5  # seconds
    sample_rate = 16000
    
    print("Recording... Speak now!")
    audio_data = sd.rec(int(duration * sample_rate), 
                       samplerate=sample_rate,
                       channels=1,
                       dtype=np.int16,
                       device=3)
    sd.wait()
    print("Recording finished!")

    # Save to WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 2 bytes for int16
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    # Initialize Groq client
    client = Groq(api_key=GROQ_API_KEY)
    
    try:
        # Send audio to Whisper API
        response = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=("audio.wav", wav_buffer.getvalue())
        )
        
        return response.text
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def describe_the_scene():
    capture_dir = Path("assets/capture")
    
    # Check if directory exists
    if not capture_dir.exists():
        print("Error: assets/capture directory not found")
        return None
    
    # Get list of capture files and sort by name (which includes timestamp)
    captures = glob.glob(str(capture_dir / "capture_*.jpg"))
    if not captures:
        print("No captures found in directory")
        return None
    
    # Get most recent file
    latest_capture = max(captures)
    
    # Encode image to base64
    with open(latest_capture, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Initialize Groq client
    client = Groq(api_key=GROQ_API_KEY)
    
    DESCRIPTION_PROMPT = """
    Describe what you see in this image in a brief, clear way. 
    Focus on the main objects and their spatial relationships.
    Keep your response under 2 sentences.
    """
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DESCRIPTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ],
                }
            ],
            model="llama-3.2-90b-vision-preview",
            temperature=0.2
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error getting scene description: {e}")
        return None

def select_policy(user_input):
    policies_info = {
        "pills-picking": "Reach for the box of my pills",
        "cup-dragging": "Bring my glass of water closer to me",
    }
    
    client = Groq(api_key=GROQ_API_KEY)

    policies_str = '\n'.join([f'- {name}: {desc}' for name, desc in policies_info.items()])
    
    prompt = f"""
        Given the following list of available robot policies and their descriptions:
        {policies_str}

        Based on the user's request: "{user_input}"
        Return ONLY the name of the most appropriate policy from the list. If none match well, return "none".
        Response should be just the policy name or "none", nothing else.
    """

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    
    selected_policy = response.choices[0].message.content.strip().lower()
    
    # Validate the response is in our policies
    if selected_policy in policies_info or selected_policy == "none":
        return selected_policy
    return "none"

if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description='Robot Control Script')
    parser.add_argument('--mode', type=str, choices=['mind', 'llama'], 
                       default='llama', help='Control mode (default: llama)')
    
    args = parser.parse_args()
    
    controller = RealRobotController(camera_id=2)
    
    run_streamlit()
    
    if args.mode == "mind":
        your_app_client_id = '7qSK1Gf5OYPKmITc8m7ek6oD4mUL3XqJ8hWVVGnK'
        your_app_client_secret = 'xNXRm4oadS3NUKv9mOhzzKhbzcW4caesGIvgi3uHaTWA9tTLLBm4WBzzUX1QKe6jLtqrhCrjAE87a3398FzS415prPUlh6cX164wE2WEjqRbrOSQQWugCQPrIVP5ccdI'

        l = LiveAdvance(your_app_client_id, your_app_client_secret)

        trained_profile_name = 'my-mental-commands' 
        l.start(trained_profile_name, controller)
    else:    
        for i in range(1):
            result = controller.llama.chat_completion()
            
            if result["action"] in ["up", "down", "rotate_left", "rotate_right"]:
                controller.handle_command(result["action"])
            else:
                print(f"Error {result} is not mapped ")
            
            time.sleep(1)            
        
        init_robot()
