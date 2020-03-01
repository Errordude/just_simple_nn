import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([4.0, 5.0, 6.0, 7.0, 8.0], dtype=float)
ys = np.array([16.0, 25.0, 36.0, 49.0, 64.0], dtype=float)
model.fit(xs, ys, epochs=5000)
print(model.predict([10.0]))