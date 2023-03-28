# Manejo de los gradientes en tensorflow

import tensorflow as tf

a = tf.Variable([2.0, 1.0])

with tf.GradientTape() as tape:
    a_squared = tf.square(a)
    a_log = tf.math.log(a_squared)

gradients = tape.gradient(a_log, a)

# Matrix operations
x_mat = tf.Variable([[0.1, 2.0, 3.5],
                     [0.5, 1.4, 5.6],
                     [0.2, 4.6, 0.8]])

const_mat = tf.constant([[0.1, 0.2, 0.3]])

with tf.GradientTape() as tape:
    mult = const_mat @ x_mat

gradients = tape.jacobian(mult, x_mat)

# Data

x = tf.constant([[1.],
                 [2.],
                 [3.],
                 [4.],
                 [5.],
                 [6.],
                 [7.],
                 [8.],
                 [9.],
                 [10.]])  # 10 x 1
y = tf.constant([[3.],
                 [5.],
                 [7.],
                 [9.],
                 [11.],
                 [13.],
                 [15.],
                 [17.],
                 [19.],
                 [21.]])  # 10 x 1

# Parameters
w = tf.Variable(tf.random.normal(shape=[1, 1]))  # 1 x 1  (weights)
b = tf.Variable(tf.zeros(shape=[1, 1]))  # 1 x 1  (bias)
print("Parámetros iniciales: ", w, b)

# Training loop

for i in range(1000):
    with tf.GradientTape() as t:
        y_pred = tf.matmul(x, w) + b  # 10 x 1
        loss = tf.reduce_mean(tf.square(y - y_pred))  # scalar
    # Gradients (backpropagation)
    w_grads, b_grads = t.gradient(loss, [w, b])
    # Gradient update (gradient descent)
    w.assign(w - tf.multiply(0.01, w_grads))
    b.assign(b - tf.multiply(0.01, b_grads))
    if i % 100 == 0:
        print("Gradientes: ", w_grads, b_grads)
        print("Parámetros actualizados: ", w, b)
        print("Func. de pérdida: ", loss)
