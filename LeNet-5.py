import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, datasets, optimizers

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

train_images = train_images/255.0
test_images = test_images/255.0

model = Sequential()
model.add(layers.Conv2D(6, (3, 3), strides=1, padding='valid', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D((2, 2), 2))
model.add(layers.LeakyReLU())
model.add(layers.Conv2D(16, (3, 3), strides=1))
model.add(layers.MaxPool2D((2, 2),strides=2))
model.add(layers.LeakyReLU())
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))

model.summary()

model.compile(optimizer=optimizers.SGD(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['acc'])

model.fit(train_images, train_labels, 32, 5, validation_split=0.1, verbose=1)

print(model.evaluate(test_images, test_labels, batch_size=32))
