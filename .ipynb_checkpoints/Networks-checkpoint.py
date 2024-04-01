import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, Concatenate

def network(x, reuse):
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.compat.v1.variable_scope('VGG_19', reuse=reuse):
        b, g, r = tf.split(x, 3, 3)
        print(b)
        print(b.shape)
        # bgr = tf.concat([b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)
        bgr = Concatenate(axis=-1)([b - VGG_MEAN[0],g - VGG_MEAN[1],r - VGG_MEAN[2]])
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        net_in = Input(shape=(224, 224, 3), name='input')(bgr)
        # Construct the network
        """conv1"""
        network = Conv2D(64, (3, 3), strides=(1, 1), activation=tf.nn.relu, padding='SAME', name='conv1_1')(net_in)
        network = Conv2D(64, (3, 3), strides=(1, 1), activation=tf.nn.relu, padding='SAME', name='conv1_2')(network)
        network = MaxPooling2D((2, 2), strides=(2, 2), padding='SAME', name='pool1')(network)
        '''conv2'''
        network = Conv2D(128, (3, 3), strides=(1, 1), activation=tf.nn.relu, padding='SAME', name='conv2_1')(network)
        network = Conv2D(128, (3, 3), strides=(1, 1), activation=tf.nn.relu, padding='SAME', name='conv2_2')(network)
        network = MaxPooling2D((2, 2), strides=(2, 2), padding='SAME', name='pool2')(network)
        '''conv3'''
        network = Conv2D(256, (3, 3), strides=(1, 1), activation=tf.nn.relu, padding='SAME', name='conv3_1')(network)
        network = Conv2D(256, (3, 3), strides=(1, 1), activation=tf.nn.relu, padding='SAME', name='conv3_2')(network)
        network = Conv2D(256, (3, 3), strides=(1, 1), activation=tf.nn.relu, padding='SAME', name='conv3_3')(network)
        network = Conv2D(256, (3, 3), strides=(1, 1), activation=tf.nn.relu, padding='SAME', name='conv3_4')(network)
        network = MaxPooling2D((2, 2), strides=(2, 2), padding='SAME', name='pool3')(network)
        '''conv4'''
        network = Conv2D(512, (3, 3), strides=(1, 1), activation=tf.nn.relu, padding='SAME', name='conv4_1')(network)
        network = Conv2D(512, (3, 3), strides=(1, 1), activation=tf.nn.relu, padding='SAME', name='conv4_2')(network)
        network = Conv2D(512, (3, 3), strides=(1, 1), activation=tf.nn.relu, padding='SAME', name='conv4_3')(network)
        network = Conv2D(512, (3, 3), strides=(1, 1), activation=tf.nn.relu, padding='SAME', name='conv4_4')(network)
        network = MaxPooling2D((2, 2), strides=(2, 2), padding='SAME', name='pool4')(network)
        '''conv5'''
        network = Conv2D(512, (3, 3), strides=(1, 1), activation=tf.nn.relu, padding='SAME', name='conv5_1')(network)
        network = Conv2D(512, (3, 3), strides=(1, 1), activation=tf.nn.relu, padding='SAME', name='conv5_2')(network)
        network = Conv2D(512, (3, 3), strides=(1, 1), activation=tf.nn.relu, padding='SAME', name='conv5_3')(network)
        network = Conv2D(512, (3, 3), strides=(1, 1), activation=tf.nn.relu, padding='SAME', name='conv5_4')(network)
        network = MaxPooling2D((2, 2), strides=(2, 2), padding='SAME', name='pool5')(network)
        conv = network
        """fc6-8"""
        network = Flatten(name='flatten')(network)
        network = Dense(4096, activation=tf.nn.relu, name='fc6')(network)
        network = Dense(4096, activation=tf.nn.relu, name='fc7')(network)
        network = Dense(1000, activation=tf.identity, name='fc8')(network)
        return network, conv


from tensorflow.keras.layers import Input, Conv2D 
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense 
from tensorflow.keras import Model
# input

# 1st Conv Block

x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 2nd Conv Block

x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 3rd Conv block

x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# 4th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

# 5th Conv block

x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
# Fully connected layers

x = Flatten()(x)
x = Dense(units = 4096, activation ='relu')(x)
x = Dense(units = 4096, activation ='relu')(x)
output = Dense(units = 1000, activation ='softmax')(x)
# creating the model

model = Model (inputs=input, outputs =output)
model.summary()