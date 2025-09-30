import cv2
import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import io

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

# Function to detect faces in uploaded image
def detect_faces_in_image(image, scale_factor, min_neighbors, rect_color_rgb):
    # Convert PIL image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(opencv_image, (x, y), (x + w, y + h), rect_color_rgb, 2)
    
    # Convert back to RGB for display
    result_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    
    return result_image, len(faces)

# Define the Streamlit app
def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    
    # Instructions
    st.markdown("""
    ### Instructions
    1. **Upload an Image**: Use the file uploader below to upload an image (JPG, JPEG, or PNG).
    2. **Adjust Parameters**: Use the sliders to set the `scaleFactor` (1.1–2.0, controls detection speed vs. accuracy) and `minNeighbors` (3–10, controls detection strictness).
    3. **Choose Rectangle Color**: Pick a color for the face detection rectangles using the color picker.
    4. **View Results**: The processed image with detected faces will be displayed below.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Parameter inputs
        st.subheader("Detection Parameters")
        scale_factor = st.slider("Scale Factor", min_value=1.1, max_value=2.0, value=1.3, step=0.1, 
                                help="Higher values make detection faster but less accurate.")
        min_neighbors = st.slider("Min Neighbors", min_value=3, max_value=10, value=5, step=1, 
                                 help="Higher values reduce false positives but may miss some faces.")
        
        # Color picker for rectangle
        rect_color_hex = st.color_picker("Rectangle Color", value="#00FF00", 
                                        help="Choose the color for face detection rectangles.")
        rect_color_rgb = hex_to_rgb(rect_color_hex)
        
        # Process the image
        if st.button("Detect Faces"):
            with st.spinner("Processing image..."):
                result_image, num_faces = detect_faces_in_image(image, scale_factor, min_neighbors, rect_color_rgb)
                
                with col2:
                    st.subheader("Processed Image")
                    st.image(result_image, caption=f"Detected {num_faces} face(s)", use_column_width=True)
                
                # Display results
                if num_faces > 0:
                    st.success(f"✅ Successfully detected {num_faces} face(s) in the image!")
                else:
                    st.info("ℹ️ No faces detected in the image. Try adjusting the parameters.")
                
                # Option to download the processed image
                result_pil = Image.fromarray(result_image)
                buf = io.BytesIO()
                result_pil.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Processed Image",
                    data=byte_im,
                    file_name=f"face_detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
    else:
        st.info("👆 Please upload an image to get started!")
        
        # Show example of what the app can do
        st.subheader("Example")
        st.markdown("""
        This app uses the **Viola-Jones algorithm** to detect faces in uploaded images. 
        
        **Features:**
        - Real-time parameter adjustment
        - Customizable detection rectangle colors
        - Downloadable results
        - Support for JPG, JPEG, and PNG formats
        
        **Tips for better results:**
        - Use clear, well-lit images
        - Ensure faces are facing forward
        - Adjust the scale factor and min neighbors based on your image
        """)

if __name__ == "__main__":
    app()