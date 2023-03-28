import tensorflow as tf


# Custom layer:

class AddRandomNoise(tf.keras.layers.Layer):

    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.n_runs = None
        self.shape_in = None

    def build(self, input_shape):
        self.shape_in = input_shape
        self.n_runs = self.add_weight(shape=(), initializer=tf.zeros_initializer(),
                                      trainable=False)

    def call(self, inputs, *args, **kwargs):
        self.n_runs.assign_add(1)
        return inputs + self.size * tf.random.normal(self.shape_in)


inputs = tf.constant([[1, 2, 4]], dtype=tf.float32)
noise_layer = AddRandomNoise(size=10)
outputs = noise_layer(inputs)
