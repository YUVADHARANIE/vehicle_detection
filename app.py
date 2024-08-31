import cv2
import streamlit as st
import numpy as np

# Function to detect cars in the video and yield frames
def detect_cars_in_video(video_path, cascade_path):
    cap = cv2.VideoCapture(video_path)
    car_cascade = cv2.CascadeClassifier(cascade_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
        
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        # Convert frame to RGB format for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb
    
    cap.release()

# Streamlit App Interface
st.title('Car Detection in Video')

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    st.text("Processing video...")
    
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_video.read())
        tfile_path = tfile.name
    
    # Detect cars in the video and display frames
    cascade_src = 'cars.xml'  # Path to the cars.xml file
    frames = detect_cars_in_video(tfile_path, cascade_src)
    
    for frame in frames:
        st.image(frame, caption='Car Detection', use_column_width=True)
