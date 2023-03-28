import tensorflow as tf


class Dense(tf.Module):
    def __init__(self, output_shape: int, name=None):
        super().__init__(name=name)
        self.built = False
        self.output_shape = output_shape

    @tf.Module.with_name_scope
    def __call__(self, x):
        if not self.built:
            self.w = tf.Variable(tf.random.normal(shape=[x.shape[1], self.output_shape]), name='w')
            self.b = tf.Variable(tf.zeros(shape=[1, self.output_shape]), name='b')
            self.built = True

        return tf.matmul(x, self.w) + self.b


class Activation(tf.Module):
    def __init__(self, activation: str = 'relu', name=None):
        super().__init__(name=name)
        self.activation = activation

    @tf.Module.with_name_scope
    def __call__(self, x):
        if self.activation == 'relu':
            return tf.nn.relu(x)
        elif self.activation == 'tanh':
            return tf.nn.tanh(x)
        elif self.activation == 'sigmoid':
            return tf.nn.sigmoid(x)
        elif self.activation == 'softmax':
            return tf.nn.softmax(x, axis=1)
        else:
            raise ValueError("Please select a valid activation function."
                             " Valid activation functions are:"
                             " `relu`, `tanh`, `sigmoid`, `softmax`.")


class NeuralNetwork(tf.Module):
    def __init__(self, layers: list[tf.Module], name=None):
        super().__init__(name=name)
        self.layers = layers
        self.loss_fn = None
        self.learning_rate = None
        self.metric = None

    def __call__(self, x):
        for layer in self.layers:
            x = layer.__call__(x)
        return x

    def add(self, layer: tf.Module):
        self.layers.append(layer)

    def compile(self, loss_fn: str, metric: str = 'accuracy', learning_rate: float = 0.01):
        if loss_fn == 'mse':
            self.loss_fn = lambda y, y_pred: tf.reduce_mean(tf.square(y - y_pred))
        elif loss_fn == 'cce':
            self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        else:
            raise ValueError("Please specify a valid loss function")

        if metric == 'mse':
            self.metric = lambda y, y_pred: tf.reduce_mean(tf.square(y - y_pred))
        elif metric == 'accuracy':
            self.metric = lambda y, y_pred: tf.reduce_mean(tf.cast(y == tf.cast(tf.argmax(y_pred, axis=1),
                                                                                dtype=tf.int32),
                                                                   dtype=tf.int32))
        else:
            raise ValueError("Please specify a valid metric")

        self.learning_rate = learning_rate

    def fit(self, x, y, epochs: int):
        history = {
            'loss': [],
            'metric': []
        }
        for i in range(epochs):
            with tf.GradientTape() as tape:
                y_pred = self(x)
                loss = self.loss_fn(y, y_pred)
            gradients = tape.gradient(loss, self.variables)
            metric = self.metric(y, y_pred)
            history['loss'].append(loss.numpy())
            history['metric'].append(metric.numpy())
            # Parameter update
            for grad, var in zip(gradients, self.variables):
                var.assign_add(-self.learning_rate * grad)
        return history

    def eval(self, x, y):
        y_pred = self(x)
        acc = self.metric(y, y_pred)
        return dict(result=y_pred, metric=acc)
