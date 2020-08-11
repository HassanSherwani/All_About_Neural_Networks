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