import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

# Load your data
x_train = np.load('/project/npy/x_train.npy')  # Assuming you have numpy arrays saved as 'train_data.npy'
y_train = np.load('/project/npy/y_train.npy')  # Assuming you have numpy arrays saved as 'train_labels.npy'
x_test = np.load('npy/x_test.npy')  # Assuming you have numpy arrays saved as 'test_data.npy'
y_test = np.load('npy/y_test.npy')  # Assuming you have numpy arrays saved as 'test_labels.npy'

# Preprocess your data (normalize, etc.)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define your model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(248, 248, 3))

# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of VGG19
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
predictions = Dense(2)(x)  # Assuming you want to predict 2 coordinates

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Save the trained model
model.save('vgg19_coordinates_model.h5')
