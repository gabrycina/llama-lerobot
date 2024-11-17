import streamlit as st
import cv2
import time
import os
from datetime import datetime
import pandas as pd

# Set page config for dark theme and wide layout
st.set_page_config(
    page_title="Robot Control Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for shadcn-like styling
st.markdown("""
    <style>
        .stApp {
            background-color: #09090b;
            color: #fafafa;
        }
        .stats-card {
            background-color: #1c1c1c;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #27272a;
        }
        .css-1d391kg {
            background-color: #1c1c1c;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'last_action' not in st.session_state:
    st.session_state.last_action = None

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.title("Robot Camera Feed")
    
    # Initialize camera
    camera = cv2.VideoCapture(2)
    if not camera.isOpened():
        st.error("Error: Could not open camera.")
        st.stop()

    # Configure camera
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Main video feed
    video_placeholder = st.empty()

with col2:
    st.title("Statistics")
    
    # Stats containers
    fps_container = st.empty()
    last_frame_container = st.empty()
    action_container = st.empty()
    
    # Last captured frame from robot
    st.subheader("Last Robot Capture")
    robot_frame_container = st.empty()

try:
    while True:
        # Update stats
        elapsed_time = time.time() - st.session_state.start_time
        current_fps = st.session_state.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Get camera frame
        ret, frame = camera.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update stats
            st.session_state.frame_count += 1
            
            # Display stats in cards
            with fps_container.container():
                st.markdown("""
                    <div class="stats-card">
                        <h3>Performance Metrics</h3>
                        <p>FPS: {:.2f}</p>
                        <p>Total Frames: {}</p>
                        <p>Uptime: {:.1f}s</p>
                    </div>
                """.format(current_fps, st.session_state.frame_count, elapsed_time), unsafe_allow_html=True)
            
            # Get last robot capture
            capture_dir = "assets/capture"
            if os.path.exists(capture_dir):
                files = [f for f in os.listdir(capture_dir) if f.endswith('.jpg')]
                if files:
                    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(capture_dir, x)))
                    latest_img = cv2.imread(os.path.join(capture_dir, latest_file))
                    if latest_img is not None:
                        latest_img_rgb = cv2.cvtColor(latest_img, cv2.COLOR_BGR2RGB)
                        robot_frame_container.image(latest_img_rgb, caption="Last Captured Frame", use_container_width=True)
        
        time.sleep(0.033)

except Exception as e:
    st.error(f"Error: {str(e)}")
finally:
    camera.release()