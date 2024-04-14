import tensorflow as tf #type:ignore

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to predict coordinates
def predict_coordinates(image):
    model = load_model("vgg19_coordinates_model_2.h5")
    prediction = model.predict(image)
    return prediction[0][0], prediction[0][1]
