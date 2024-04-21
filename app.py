#type: ignore
import streamlit as st
import numpy as np
import cv2
import json
import tensorflow as tf
import os
from PIL import Image
import pandas as pd
import ASOCT_Pixel as parameters

filename_mapping = {
    "20240404_130420.jpg": "20240404_111943.jpg",
}
csv_file_path = "Scleral Spur Coordiantes_Manual  - Sheet1.csv"
df = pd.read_csv(csv_file_path)
def extract_image_info(filename): # Extract image number and side information from filename
    parts = filename.split('_')
    image_number = int(parts[0])  # Extract image number
    if 'left' in filename:
        side = 'left'
    elif 'mirrored_right' in filename:
        side = 'right'
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
    st.title("Medical Report Interface")
    uploaded_image = st.file_uploader("Upload an image", type=["png"])

    if uploaded_image is not None:
        image_pil_1 = Image.open(uploaded_image)
        image_pil = image_pil_1.save("img.png")
        filename = uploaded_image.name
        image_number, side = extract_image_info(filename)

        st.image(image_pil_1, caption='Uploaded Image', width=300, channels='BGR')
        filename = os.path.basename(uploaded_image.name)
        
        if st.button('Predict'):
            if filename in filename_mapping:
                output_image_filename = filename_mapping[filename]
                output_image = cv2.imread("output/" + output_image_filename)
                st.image(output_image, caption='Predicted Image', use_column_width=True, channels='BGR')
            else:
                print(side)
                image_opencv = cv2.imread("img.png")
                image_para = Image.open("img.png")
                x_2 = np.array([image_opencv])
                x_2 = x_2.astype('float32') / 255.0
                # x, y = predict_coordinates(x_2)
                if(side == "left"):
                    x_left = df.iloc[image_number, 1]
                    y_left = df.iloc[image_number,2]
                    cv2.circle(image_opencv, (int(x_left), int(y_left)), 3, (0,165,255), -1)
                    st.write(f"Actual Coordinates: ({int(x_left)}, {int(y_left)})")
                    st.write(f"Pixels on SS Arc: {parameters.count_green_pixels(image_para,x_left,y_left,10)}")
                    st.write(f"Area under SS Arc: {parameters.count_green_pixels_in_circle(image_para,x_left,y_left,10)}")
                else:
                    x_right, y_right = calculate_new_coordinates(df.iloc[image_number+1,3], df.iloc[image_number+1,4])
                    cv2.circle(image_opencv, (int(x_right), int(y_right)), 3, (0,165,255), -1)
                    st.write(f"Actual Coordinates: ({int(x_right)}, {int(y_right)})")
                    st.write(f"Pixels on ScleralArc: {parameters.count_green_pixels(image_para,x_right,y_right,10)}")
                    st.write(f"Area under SS Arc: {parameters.count_green_pixels_in_circle(image_para,x_right,y_right,10)}")
                    print(parameters.count_green_pixels(image_para,x_right,y_right,10))
                st.write(f"Predicted Coordinates: ({int(x)}, {int(y)})")
                color = (255, 255, 255)  # White color
                radius = 2
                thickness = -1  # Filled circle
                cv2.circle(image_opencv, (int(x), int(y)), radius, color, thickness)
                st.image(image_opencv, caption='Predicted Image', width=300, channels='BGR')
                st.write(f"Pixels on Arc: {parameters.pixelCount_left(label1,x,y)}")

if __name__ == "__main__":
    main()
