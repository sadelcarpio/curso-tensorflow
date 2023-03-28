# Operaciones en tensores

import tensorflow as tf

a = tf.constant([i for i in range(12)], shape=[3, 4], dtype=tf.float32)
b = tf.constant([i for i in range(12)], shape=[4, 3], dtype=tf.float32)
b_t = tf.transpose(b)
c = tf.random.normal([3, 4, 3])
x_im = tf.constant([i for i in range(16)], shape=[1, 4, 4, 1], dtype=tf.float32)  # 1, 4, 4, 1
filters = tf.constant(  # 2, 2, 1, 1
    [[[[0.2]], [[0.5]]],
     [[[0.1]], [[0.2]]]]
)

# Algebraic ops (elem by elem)
addition = tf.add(a, b_t)
mult = tf.multiply(a, b_t)
sub = tf.subtract(a, b_t)
power = tf.pow(a, 3)  # a ** 3

# Agregation ops
c_sum = tf.reduce_sum(c, axis=0)
c_mean = tf.reduce_mean(c, axis=0)

# Tensor ops
y = tf.matmul(a, b)
y_batch = tf.matmul(a, c)  # 3, 3, 3
conv = tf.nn.conv2d(x_im, filters, strides=[1, 1], padding='VALID')
