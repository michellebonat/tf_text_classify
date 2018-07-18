import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

# Download IMDB dataset of 50k movie reviews
# num_words=10000 keeps only the top 10,000 most frequently occurring words in the training data
imdbimdb  ==  keraskeras..datasetsdataset .imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)