# 1) Import libraries and modules
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

# 2) Load data from remote url

dataset_url = 'https://s3.amazonaws.com/worldquant-dropbox/datasimply_export.csv'
data = pd.read_csv(dataset_url)

# Uncomment this to print first 5 rows of data
# print data.head()

# Uncomment this to print the data shape
# print data.shape

# Uncomment this to print data overview summary stats
print data.describe()