#type: ignore
import streamlit as st
import numpy as np
import cv2
import json
import os
from PIL import Image
import tensorflow as tf
import pandas as pd

filename_mapping = {
    "20240404_130420.jpg": "20240404_111943.jpg",
}
csv_file_path = "Scleral Spur Coordiantes_Manual  - Sheet1.csv"
df = pd.read_csv(csv_file_path)
def extract_image_info(filename):
    # Extract image number and side information from filename
    parts = filename.split('_')
    image_number = int(parts[0])  # Extract image number
    side = parts[-1]  # Extract side information (left_half or mirrored_right_half)
    return image_number, side

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to predict coordinates
def predict_coordinates(image):
    model = load_model("vgg19_coordinates_model_2.h5")
    prediction = model.predict(image)
    return prediction[0][0], prediction[0][1]

def calculate_new_coordinates(original_x, original_y):
    width = 496
    middle = width // 2
    if original_x < middle:
        return [original_x, original_y]
    else:
        new_x = width - original_x - 1
        return [new_x, original_y]
# Main Streamlit app
def main():
    st.title("Accurate localization of key points Scleral Spur based on VGG19 on ACA and measurement of relevant")
    uploaded_image = st.file_uploader("Upload an image", type=["png"])
    filename = uploaded_image.name
    image_number, side = extract_image_info(filename)

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
                if(side == "left"):
                    x_1 = df.iloc[image_number, 1]
                    y_1 = df.iloc[image_number,2]
                    cv2.circle(image_opencv, (int(x_1), int(y_1)), 2, (255,0,255), -1)
                else:
                    x_1 = df.iloc[image_number+1, 1]
                    y_1 = df.iloc[image_number+1,2]
                    cv2.circle(image_opencv, (int(x_1), int(y_1)), 2, (255,0,255), -1)
                st.write(f"Predicted Coordinates: ({x}, {y})")
                color = (255, 255, 255)  # White color
                radius = 2
                thickness = -1  # Filled circle
                cv2.circle(image_opencv, (int(x), int(y)), radius, color, thickness)
                st.image(image_opencv, caption='Predicted Image', width=300, channels='BGR')

if __name__ == "__main__":
    main()
