import tensorflow as tf
from tensorflow import keras

#set up model
model = tf.keras.Sequential ([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',
                            input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

#generates 64 filters and multiplies each of them 
#across the image; during epochs the NN will decide
#which filter gave the best signals to helps 
#match images to thier labels

#can stack conv layers