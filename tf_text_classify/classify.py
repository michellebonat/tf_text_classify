import tensorflow as tf
from tensorflow import keras

import numpy as np

# print(tf.__version__)

# Download the IMDB dataset, keeping the top 10,000 most frequently occurring words in the training data
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Explore the data
# Training entries
# print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# The first review as integers
# print(train_data[0])

# See how many words in the first and second reviews
# Note: inputs to a neural network must be the same length, we'll need to resolve this later
# len(train_data[0]), len(train_data[1])

# Convert integers back to words
# Creates a helper function to query a dictionary object that contains the integer to string mapping
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Use the decode_review function to display the text for the first review
decode_review(train_data[0])

# Prepare the data
# Convert the reviews (arrays of integers) into tensors to be fed into neural network using pad_sequences
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# Look at the length of the examples now, they should be the same as each other
len(train_data[0]), len(train_data[1])

# Look at the first review again, it is now padded, standardized length
print(train_data[0])

# Build the model
# Input data consists of array of word-indices. The labels to predict are either 0 or 1
# Input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# Configure the model to use a loss function and optimizer
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Create a validation set from the training data
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Train the model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# Evaluate the model
# results = model.evaluate(test_data, test_labels)

# print(results)

# Create a graph of accuracy and loss over time
# Dictionary with an entry for each of the four metrics monitored during training and validation
history_dict = history.history
history_dict.keys()

# Plot the training and validation loss for comparison, as well as the training and validation accuracy
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Plot training and validation loss
# "bo" is for "blue dot"
# plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()

# Plot training and validation accuracy
plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()