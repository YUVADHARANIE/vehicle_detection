import cv2
import tempfile
import streamlit as st
import numpy as np
import os

# Function to detect cars in the video
def detect_cars_in_video(video_path, cascade_path):
    cap = cv2.VideoCapture(video_path)
    car_cascade = cv2.CascadeClassifier(cascade_path)
    
    if not cap.isOpened():
        st.error("Error opening video file.")
        return None
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create a temporary file to save the processed video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file_path = temp_file.name
    temp_file.close()
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(temp_file_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
        
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        video_writer.write(frame)
    
    cap.release()
    video_writer.release()
    
    return temp_file_path

# Streamlit App Interface
st.title('Car Detection in Video')

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    st.text("Processing video...")
    
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_video.read())
        tfile_path = tfile.name
    
    # Detect cars in the video
    cascade_src = 'cars.xml'  # Path to the cars.xml file
    processed_video_path = detect_cars_in_video(tfile_path, cascade_src)
    
    if processed_video_path:
        # Display the processed video in the app
        st.text("Car detection completed. Displaying video below:")
        st.video(processed_video_path)
    else:
        st.error("Failed to process the video.")
