import tensorflow as tf

# Data: MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Data preprocessing
x_train = x_train / 255
x_test = x_test / 255

# Model declaration (Sequential API)

# Flatten and Dense
neural_network = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

neural_network.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

nn_history = neural_network.fit(x_train, y_train, validation_data=(x_test, y_test), shuffle=True,
                                batch_size=128, epochs=10)

# LeNet (CNN)

# Preprocessing for LeNet (shape = [N, H, W, C])
x_train = tf.reshape(x_train, [-1, 28, 28, 1])
x_test = tf.reshape(x_test, [-1, 28, 28, 1])

lenet = tf.keras.models.Sequential()
lenet.add(tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu'))
lenet.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
lenet.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'))
lenet.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
lenet.add(tf.keras.layers.Flatten())
lenet.add(tf.keras.layers.Dense(120, activation='relu'))
lenet.add(tf.keras.layers.Dense(84, activation='relu'))
lenet.add(tf.keras.layers.Dense(10, activation='softmax'))

lenet.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

lenet_history = lenet.fit(x_train, y_train, validation_data=(x_test, y_test), shuffle=True,
                          batch_size=128, epochs=10)
