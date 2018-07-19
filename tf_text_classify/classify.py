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