# Visualización de operaciones tensoriales como un grafo dirigido (DAG)

import tensorflow as tf

logs_dir = './logs'

graph = tf.Graph()

with graph.as_default():
    tensor_a = tf.constant([[1, 2], [0.5, 1], [2.5, 3]], name='a')  # 3 x 2
    tensor_b = tf.constant([[5, 6, 7.0], [2.4, 1, 4]], name='b')  # 2 x 3
    tensor_c = tf.matmul(tensor_a, tensor_b, name='mult')  # 3 x 3
    tensor_d = tf.constant([[1., 5., 9.]], name='d')  # 1 x 3
    tensor_e = tf.add(tensor_d, tensor_c, name='add')  # 3 x 3
    tensor_f = tf.reduce_sum(tensor_e, axis=0, name='sum', keepdims=True)  # 1 x 3

sess = tf.compat.v1.Session(graph=graph)

print(sess.run([tensor_e, tensor_f]))

writer = tf.summary.create_file_writer(logs_dir)

with writer.as_default():
    tf.summary.graph(graph)
