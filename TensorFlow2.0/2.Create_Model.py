#Loading and exploring data

#1- Importing key modules
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Load data
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
#print(len(train_labels))
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#plt.imshow(train_images[7])
#plt.show()
# we will define a list of the class names and pre-process images. We do this by dividing each image by 255.
# Since each image is greyscale we are simply scaling the pixel values down to make computations easier for our model
train_images = train_images/255.0
test_images = test_images/255.0

# Building the model

#a. Creating model i.e flatten the data

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")
	])

#b.Training the Model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) # use spare_categorical_crossenropy for multi-label problems

model.fit(train_images, train_labels, epochs=5)

#c. Testing model

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)

#Save the model

