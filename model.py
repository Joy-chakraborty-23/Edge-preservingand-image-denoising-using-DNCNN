import tensorflow as tf
from tensorflow.keras import layers

def build_model(input_shape=(9, 9, 3)):
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', 
                      kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02),
                      input_shape=input_shape),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                      kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                      kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02)),
        layers.MaxPooling2D((3, 3), strides=3),
        layers.Flatten(),
        layers.Dense(256, activation='relu',
                     kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.02)),
        layers.Dense(2, activation='softmax')
    ])
    return model
