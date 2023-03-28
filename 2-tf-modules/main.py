import tensorflow as tf
from neural_networks import Dense, NeuralNetwork

# Data
x_train = tf.constant([[1.],
                       [2.],
                       [3.]])

y_train = tf.constant([[1.],
                       [2.],
                       [3.]])

x_val, y_val = tf.constant([[4.]]), tf.constant([[4.]])

# Model definition
model = NeuralNetwork([
    Dense(1, name='dense_layer'),
])

# Model training
model.compile(loss_fn='mse', metric='mse', learning_rate=0.1)
history = model.fit(x_train, y_train, epochs=100)

# Model evaluation
results = model.eval(x_val, y_val)

# Model inference
prediction = model(tf.constant([[20.], [21.]]))
