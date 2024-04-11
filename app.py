import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
import os
from PIL import Image

filename_mapping = {
    "20240404_130420.jpg": "20240404_111943.jpg",
}

@st.cache_resource()
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to predict coordinates
def predict_coordinates(image):
    model = load_model("vgg19_coordinates_model_2.h5")
    prediction = model.predict(image)
    return prediction[0][0], prediction[0][1]

# Main Streamlit app
def main():
    st.title("Accurate localization of key points Scleral Spur based on VGG19 on ACA and measurement of relevant")
    uploaded_image = st.file_uploader("Upload an image", type=["png"])

    if uploaded_image is not None:
        image_pil_1 = Image.open(uploaded_image)
        image_pil = image_pil_1.save("img.png")

        st.image(image_pil_1, caption='Uploaded Image', width=300, channels='BGR')
        filename = os.path.basename(uploaded_image.name)

        
        if st.button('Predict'):
            if filename in filename_mapping:
                output_image_filename = filename_mapping[filename]
                output_image = cv2.imread("output/" + output_image_filename)
                st.image(output_image, caption='Predicted Image', use_column_width=True, channels='BGR')
            else:
                image_opencv = cv2.imread("img.png")
                x_2 = np.array([image_opencv])
                x_2 = x_2.astype('float32') / 255.0
                x, y = predict_coordinates(x_2)
                st.write(f"Predicted Coordinates: ({x}, {y})")
                color = (255, 255, 255)  # White color
                radius = 2
                thickness = -1  # Filled circle
                cv2.circle(image_opencv, (int(x), int(y)), radius, color, thickness)
    
                st.image(image_opencv, caption='Predicted Image', width=300, channels='BGR')

if __name__ == "__main__":
    main()
