import cv2
import streamlit as st
import os
from datetime import datetime

# Load the face cascade classifier
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        st.error("Error: Could not load the face cascade classifier. Please ensure 'haarcascade_frontalface_default.xml' is in the correct directory.")
        st.stop()
except Exception as e:
    st.error(f"Error loading cascade classifier: {e}")
    st.stop()

# Function to convert hex color to RGB tuple
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Function to detect faces
def detect_faces(scale_factor, min_neighbors, rect_color_rgb, save_images):
    # Initialize the webcam
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not access the webcam. Please ensure it is connected and accessible.")
            return
    except Exception as e:
        st.error(f"Error accessing webcam: {e}")
        return

    # Create a directory for saved images if it doesn't exist
    if save_images:
        os.makedirs("detected_faces", exist_ok=True)

    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame from webcam.")
            break

        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color_rgb, 2)
        
        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)

        # Save the frame with detected faces if enabled
        if save_images and len(faces) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"detected_faces/face_{timestamp}.jpg"
            cv2.imwrite(save_path, frame)
            st.success(f"Image saved as {save_path}")

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Define the Streamlit app
def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    
    # Instructions
    st.markdown("""
    ### Instructions
    1. **Adjust Parameters**: Use the sliders below to set the `scaleFactor` (1.1–2.0, controls detection speed vs. accuracy) and `minNeighbors` (3–10, controls detection strictness).
    2. **Choose Rectangle Color**: Pick a color for the face detection rectangles using the color picker.
    3. **Enable Image Saving**: Check the box to save images with detected faces to a 'detected_faces' folder.
    4. **Start Detection**: Click the "Detect Faces" button to begin webcam-based face detection.
    5. **Stop Detection**: Press the 'q' key in the webcam window to stop.
    """)

    # Parameter inputs
    st.subheader("Detection Parameters")
    scale_factor = st.slider("Scale Factor", min_value=1.1, max_value=2.0, value=1.3, step=0.1, help="Higher values make detection faster but less accurate.")
    min_neighbors = st.slider("Min Neighbors", min_value=3, max_value=10, value=5, step=1, help="Higher values reduce false positives but may miss some faces.")
    
    # Color picker for rectangle
    rect_color_hex = st.color_picker("Rectangle Color", value="#00FF00", help="Choose the color for face detection rectangles.")
    rect_color_rgb = hex_to_rgb(rect_color_hex)

    # Checkbox for saving images
    save_images = st.checkbox("Save Images with Detected Faces", value=False, help="Saves images to a 'detected_faces' folder when faces are detected.")

    # Button to start detecting faces
    if st.button("Detect Faces"):
        detect_faces(scale_factor, min_neighbors, rect_color_rgb, save_images)

if __name__ == "__main__":
    app()