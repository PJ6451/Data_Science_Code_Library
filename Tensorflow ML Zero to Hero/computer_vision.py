import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels), (test_images,test_labels) = fashion_mnist.load_data()


#set up model
model = keras.Sequential ([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
#activation functions
#relu: rectified linear unit, returns number if greater than 0
#softmax: probability that we're looking at whatever categorical thing

#compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss ='sparse_categorical_crossentropy')
#optimizer optimizes the fucntion parameters of the middle layer

#train model
model.fit(train_images,train_labels,epochs=5)

#test model accuracy
model.evaluate(test_images,test_labels)