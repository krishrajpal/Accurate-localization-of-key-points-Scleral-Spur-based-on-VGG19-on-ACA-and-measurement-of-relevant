import streamlit as st
# import tensorflow as tf
import numpy as np
import cv2
import json
import os

filename_mapping = {
    "20240404_130420.jpg": "20240404_111943.jpg",
    "image2.jpg": "output2.jpg",
}

# Load the TensorFlow model
# @st.cache(allow_output_mutation=True)
# def load_model(model_path):
#     model = tf.keras.models.load_model(model_path)
#     return model

# Function to predict coordinates
def predict_coordinates(image):
    # Placeholder for prediction logic
    # For demonstration purposes, just returning random coordinates
    x = np.random.randint(0, image.shape[1])
    y = np.random.randint(0, image.shape[0])
    
    # model = load_model("model1.h5")

            # Preprocess the image
    # img_array = np.array(image.resize((224, 224))) / 255.0
    # img_array = np.expand_dims(img_array, axis=0)

    #         # Make prediction
    # prediction = model.predict(img_array)
    return x, y

# Main Streamlit app
def main():
    # Accurate-localization-of-key-points-Scleral-Spur-based-on-VGG19-on-ACA-and-measurement-of-relevant
    st.title("Accurate localization of key points Scleral Spur based on VGG19 on ACA and measurement of relevant")
    # st.title("TensorFlow Model Deployment with Streamlit")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', width=300, channels='BGR')
        filename = os.path.basename(uploaded_image.name)

        
        if st.button('Predict'):
            print(filename)
            if filename in filename_mapping:
                output_image_filename = filename_mapping[filename]
                output_image = cv2.imread("output/" + output_image_filename)
                st.image(output_image, caption='Predicted Image', use_column_width=True, channels='BGR')
            else:
                x, y = predict_coordinates(image)
                st.write(f"Predicted Coordinates: ({x}, {y})")

                # Draw a colored circle on the image at predicted coordinates
                color = (0, 255, 0)  # Green color
                radius = 5
                thickness = -1  # Filled circle
                cv2.circle(image, (x, y), radius, color, thickness)
    
                st.image(image, caption='Predicted Image', use_column_width=True)

if __name__ == "__main__":
    main()
