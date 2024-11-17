from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config
from lerobot.common.robot_devices.control_utils import init_policy, control_loop

import pyttsx3
import base64
import cv2

from langchain_core.tools import tool

from langchain.schema import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from dotenv import load_dotenv
import os
import sounddevice as sd
import numpy as np
from groq import Groq
import wave
import io

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)

# models = {
#     "grab_sponge":  {"repo_id": "1g0rrr/grab_sponge", "control_time_s": 32},
#      "grab_orange": {"repo_id": "1g0rrr/grab_orange", "control_time_s": 10}, 
#      "grab_candy":{"repo_id": "1g0rrr/grab_candy", "control_time_s": 10}
# }


def do_control_loop(policy_obj, robot,models, policies, display_cameras = True):
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
    robot_cfg = init_hydra_config("../lerobot/configs/robot/moss.yaml", []) # la lista rappresenta robot_overrides
    robot = make_robot(robot_cfg)
    robot.connect()

    models = {
        "policy_name": {"repo_id": "user/model_name", "control_time_s": 0}
    }
    policies = {}
   
    for model_name in models:
        model = models[model_name]
        policy_overrides = ["device=cpu"] # dicono loro
        policy, policy_fps, device, use_amp = init_policy(model["repo_id"], policy_overrides)
        policies[model_name] = ({"policy": policy, "policy_fps": policy_fps, "device": device, "use_amp": use_amp, "control_time_s": model["control_time_s"]})

def listen_to_user():
    # Audio recording parameters
    duration = 5  # seconds
    sample_rate = 16000
    
    print("Recording... Speak now!")
    audio_data = sd.rec(int(duration * sample_rate), 
                       samplerate=sample_rate,
                       channels=1,
                       dtype=np.int16)
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

def select_policy(user_input):
    policies_info = {
        "take_my_pills": "Take my pills closer to me",
        "glass_of_water": "Get me a glass of water",
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
    # engine = pyttsx3.init()
    # engine.say("Dai forza voglio sti cazzo di 25k")
    # engine.runAndWait()
    transcript = listen_to_user()
    print(transcript)
    print(select_policy(transcript))
        
