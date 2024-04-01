import tensorflow as tf
import numpy as np
import cv2
import glob
import random
from Net import network  # Assuming this is your network architecture
import os
from _read_data import read_train_data, read_test_data

lr_init = 1e-5
batch_size = 64

# Define your network inputs
x = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32)
y = tf.keras.Input(shape=(2,), dtype=tf.float32)

# Build your network
net_vgg, conv = network(x)
ft_output = tf.keras.layers.Flatten(name='flatten_1')(conv)
ft_output = tf.keras.layers.Dense(4096, activation=tf.nn.relu, name='fc6_1')(ft_output)
ft_output = tf.keras.layers.Dense(4096, activation=tf.nn.relu, name='fc7_1')(ft_output)
ft_output = tf.keras.layers.Dense(2, activation=None, name='fc8_1')(ft_output)

# Define loss
mse_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, ft_output))

# Define optimizer
optimizer = tf.keras.optimizers.Adam(lr_init, beta_1=0.9)

# Define metrics
correct_pred = tf.sqrt(tf.reduce_sum(tf.square(y - ft_output), axis=1)) <= 15
accur = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Model for training
model_train = tf.keras.Model(inputs=[x, y], outputs=[ft_output, mse_loss, accur])

# Model for testing
model_test = tf.keras.Model(inputs=x, outputs=ft_output)

# Compile the training model
model_train.compile(optimizer=optimizer, loss=[None, mse_loss, None], metrics=[None, None, accur])

# Train the model
train_vec_x, train_y = read_train_data()  # Load training data
# Assuming you have the test data loader as well
test_x, test_y = read_test_data()  # Load test data

# Assuming you have a loop for training epochs and batches
for e in range(500):
    for i in range(len(train_vec_x) // batch_size):
        batch_x = train_vec_x[i * batch_size:(i + 1) * batch_size]
        batch_y = train_y[i * batch_size:(i + 1) * batch_size]
        model_train.train_on_batch([batch_x, batch_y], [None, None, None])  # Train on batch

        if i % 10 == 0:
            _, _, accur_train = model_train.evaluate([batch_x, batch_y], [None, None, accur])  # Evaluate training accuracy
            accur_test = model_train.evaluate(test_x, test_y)  # Evaluate test accuracy
            print('epoch', e, 'iter', i, 'train_accuracy:', accur_train, 'test_accuracy:', accur_test)

    # Save the model every 20 epochs
    if e % 20 == 0:
        model_train.save_weights('./model/latest')

# Alternatively, if you want to test the model
if __name__ == '__main__':
    show_num = 10
    test_x, test_y = read_test_data()
    test_x_show = test_x[0:show_num]
    test_y_show = test_y[0:show_num]
    _ft_output = model_test.predict(test_x_show)
    accur_test = tf.reduce_mean(tf.cast(tf.sqrt(tf.reduce_sum(tf.square(test_y_show - _ft_output), axis=1)) <= 15, tf.float32))
    print('test_accuracy:', accur_test)
    for i in range(show_num):
        clone_img_1 = test_x_show[i].copy()
        cv2.circle(clone_img_1, (_ft_output[i, 0], _ft_output[i, 1]), 3, (0, 0, 255), -1)
        cv2.circle(clone_img_1, (test_y[i, 0], test_y[i, 1]), 3, (0, 255, 0), -1)
        cv2.imshow('img', clone_img_1)
        cv2.waitKey(0)



