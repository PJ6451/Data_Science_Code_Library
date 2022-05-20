import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units = 1, input_shape=[1])])

model.compile(optimizer='sgd',loss='mean_squared_error')

xs = np.array([-1.,0.,1.,2.,3.,4.], dtype=float)
ys = np.array([-3.,-1.,4.,7.,10.,13.], dtype=float)

model.fit(xs,ys,epochs=500)

print(model.predict([10.]))