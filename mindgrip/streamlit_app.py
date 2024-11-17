import streamlit as st
import cv2
import os
import time

st.title("Robot Camera Feed")

# Create a placeholder for the image
frame_placeholder = st.empty()

def get_last_frame():
    """Get the last frame from the assets/capture directory"""
    try:
        capture_dir = "assets/capture"
        files = [f for f in os.listdir(capture_dir) if f.endswith('.jpg')]
        if not files:
            return None
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(capture_dir, x)))
        img_path = os.path.join(capture_dir, latest_file)
        return cv2.imread(img_path)
    except Exception as e:
        st.error(f"Error reading frame: {e}")
        return None

# Main loop
while True:
    frame = get_last_frame()
    if frame is not None:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Display the frame
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    # Small delay
    time.sleep(0.1)