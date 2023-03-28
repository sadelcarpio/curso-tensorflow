import tensorflow as tf
from houses_dataset_preprocessing import load_houses_data, preprocess_numerical_data, load_images
from sklearn.model_selection import train_test_split

# Data loading and processing
houses_info = load_houses_data(r'./Houses-dataset/Houses Dataset/HousesInfo.txt')
images = load_images(houses_info, r'./Houses-dataset/Houses Dataset')
images = images / 255
houses_info = preprocess_numerical_data(houses_info)

# Split data in train and test
(train_features, test_features, train_imgs, test_imgs) = train_test_split(houses_info, images, test_size=0.2,
                                                                          random_state=42)

train_features_x = train_features.drop('price', axis=1).to_numpy()
y_train = train_features['price'].to_numpy()

test_features_x = test_features.drop('price', axis=1).to_numpy()
y_test = test_features['price'].to_numpy()

# CNN Model (for image data)
cnn_input = tf.keras.layers.Input(shape=(64, 64, 3))
x = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu')(cnn_input)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.GlobalAvgPool2D()(x)
cnn_output = tf.keras.layers.Dense(5, activation='relu')(x)

cnn_model = tf.keras.Model(cnn_input, cnn_output)

# NN Model (for continuous data)
features_input = tf.keras.layers.Input(shape=(3,))
y = tf.keras.layers.Dense(32, activation='relu')(features_input)
y = tf.keras.layers.Dropout(0.1)(y)
features_output = tf.keras.layers.Dense(5, activation='relu')(y)
features_model = tf.keras.Model(features_input, features_output)

# Feature concatenation
concat_input = tf.keras.layers.concatenate([cnn_model.output, features_model.output])
z = tf.keras.layers.Dense(10, activation='relu')(concat_input)
output = tf.keras.layers.Dense(1, activation='linear')(z)
final_model = tf.keras.Model(inputs=[cnn_model.input, features_model.input], outputs=output)

# Model compile and train
final_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanSquaredError(),
             tf.keras.metrics.MeanAbsoluteError(),
             tf.keras.metrics.MeanAbsolutePercentageError()]
)

history = final_model.fit([train_imgs, train_features_x], y_train,
                          validation_data=([test_imgs, test_features_x], y_test),
                          batch_size=32,
                          epochs=10,
                          shuffle=True)
