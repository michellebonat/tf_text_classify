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
# The first review
print(train_data[0])
