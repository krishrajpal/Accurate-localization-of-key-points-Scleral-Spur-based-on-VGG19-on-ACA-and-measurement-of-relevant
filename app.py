import streamlit as st
# import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TensorFlow model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def main():
    st.title("TensorFlow Model Deployment with Streamlit")

    # Sidebar for file upload and model selection
    st.sidebar.title("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    st.sidebar.title("Select Model")
    model_name = st.sidebar.selectbox("Choose a model", ["Model 1", "Model 2"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make predictions
        if st.button("Make Prediction"):
            st.write("Making prediction...")

            # Load the selected model
            if model_name == "Model 1":
                model_path = "model1.h5"
            elif model_name == "Model 2":
                model_path = "model2.h5"

            model = load_model(model_path)

            # Preprocess the image
            img_array = np.array(image.resize((224, 224))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array)

            # Display prediction
            st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
