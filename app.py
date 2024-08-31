import cv2
import tempfile
import streamlit as st
from PIL import Image
import numpy as np

# Function to detect cars in the video
def detect_cars_in_video(video_path, cascade_path):
    cap = cv2.VideoCapture(video_path)
    car_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Create a temporary file to save the processed video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_writer = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
        
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_file.name, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        
        video_writer.write(frame)
    
    cap.release()
    video_writer.release()
    
    return temp_file.name

# Streamlit App Interface
st.title('Car Detection in Video')

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    st.text("Processing video...")
    
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    # Detect cars in the video
    cascade_src = 'cars.xml'  # Path to the cars.xml file
    processed_video_path = detect_cars_in_video(tfile.name, cascade_src)
    
    # Display the processed video in the app
    st.text("Car detection completed. Displaying video below:")
    video_file = open(processed_video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    
    video_file.close()
