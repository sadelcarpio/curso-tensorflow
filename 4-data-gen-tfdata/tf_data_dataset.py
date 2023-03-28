import tensorflow as tf
from matplotlib import pyplot as plt

train_dir = './butterfly-images40-species/train'

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=32,
    image_size=(224, 224)
)

class_names = train_dataset.class_names

data_augmentation = tf.keras.Sequential(
    [
     tf.keras.layers.RandomFlip('horizontal'),
     tf.keras.layers.RandomRotation(0.2),  # rotación en el rango [-0.2*2*pi, 0.2*2*pi]
    ]
)

list_dataset = tf.data.Dataset.list_files('./butterfly-images40-species/train/*/*.jpg')

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    first_image = images[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
plt.show()
