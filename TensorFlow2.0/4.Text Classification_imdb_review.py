#IMBD case for Text Classification

#1- Import key Modules
import tensorflow as tf
from tensorflow import keras
import numpy
#2- Data Loading and preparing

imdb = keras.datasets.imdb
# train-test split
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("Shape of feature train data",train_data.shape)
print("shape of label train data",train_labels.shape)
print("Shape of feature test data",test_data.shape)
print("Shape of label test data",test_labels.shape)
# 3- Integer Encoded Data

# A dictionary mapping words to an integer index
_word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])

# this function will return the decoded (human readable) reviews

# 4- Preprocess the Data

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)
print("Shape of Training data after padding",train_data.shape)
print("shape of Testing data after padding",test_data.shape)
# 5- Define the Model

model = keras.Sequential()
model.add(keras.layers.Embedding(80000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# 6-Train Model

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# 7- evaluate Model

results = model.evaluate(test_data, test_labels)
print("Probability Distribution of output:",results)

#8- Save model
model.save("model.h5")  # name it whatever you want but end with .h5
# 9- Make Prediction with new data

# load saved model
model = keras.models.load_model("model.h5")

def review_encode(s):
	encoded = [11]

	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded

with open("test.txt", encoding="utf-8") as f:
	for line in f.readlines():
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
		predict = model.predict(encode)
		print(line)
		print(encode)
		print(predict[0])