import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


training_dir = './butterfly-images40-species/train'
train_datagen = ImageDataGenerator(rescale=1/255,
                                   rotation_range=30,
                                   horizontal_flip=True,
                                   brightness_range=[0.5, 1.5])
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(224, 224),
    class_mode='sparse',
    batch_size=32
)

for _ in range(5):
    img, label = train_generator.next()
    print(img.shape)
    plt.imshow(img[0].squeeze())
    plt.title(label[0])
    plt.show()
