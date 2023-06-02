'''

The code in this file is sourced from some of the sample code provided in the competition brief on kaggle. This sample code 
aims to complete thte competition with used of a random forest. The page contanining te sample solution is called:
"Student Performance w/ TensorFlow Decision Forests"
The borrowed code is that to read the csv and load into the environment. 

This code is implemented under the pre-condition that the data file-path be kept in the same format as that specified in 
the example. The provide datafilepath is the following:

Data Explorer
4.74GB
    jo_wilder (...) - contains solution files
    jo_wilder_310 (...) - conatins another solution's files
    Data (^)
        sample_submission.csv
        test.csv
        train.csv
        train_labels.csv

'''


import tensorflow as tf
import tensorflow_addons as tfa
#import tensorflow_decision_forests as tfdf

import pandas as pd
from clean_df import CleanDataFrame
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd, numpy as np, gc
from sklearn.model_selection import KFold, GroupKFold
#from xgboost import XGBClassifier
from sklearn.metrics import f1_score

#show versions of the APIs
#print("TensorFlow Decision Forests v" + tfdf.__version__)
print("TensorFlow Addons v" + tfa.__version__)
print("TensorFlow v" + tf.__version__)


#Load the dataset and print the shape and first 5 samples

    # Reference: https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/384359
dtypes={
    'elapsed_time':np.int32,
    'event_name':'category',
    'name':'category',
    'level':np.uint8,
    'room_coor_x':np.float32,
    'room_coor_y':np.float32,
    'screen_coor_x':np.float32,
    'screen_coor_y':np.float32,
    'hover_duration':np.float32,
    'text':'category',
    'fqid':'category',
    'room_fqid':'category',
    'text_fqid':'category',
    'fullscreen':'category',
    'hq':'category',
    'music':'category',
    'level_group':'category'}

y_dtypes={
    'correct': 'category'}





# Load the CSV file into a pandas DataFrame
dataset_df = pd.read_csv('Data/train.csv', dtype=dtypes)
dataset_y = pd.read_csv('Data/train_labels.csv', dtype=y_dtypes)
dataset_df.info()
dataset_y.info()

# Print the original shape of the DataFrame
print("Full train dataset shape is {}".format(dataset_df.shape))

# Drop the 'level_group' column
dataset_df = dataset_df.drop('level_group', axis=1)
# Drop the 'index' column
#dataset_df = dataset_df.drop('index', axis=1)

# Reduces memory usage of the data
cdf = CleanDataFrame(dataset_df, max_num_cat=20)
cdf.clean(min_missing_ratio=1, drop_nan=False)
cdf.optimize()
cdf.df.info()
dataset_y.info()

# Print the new shape of the DataFrame
print("New train dataset shape is {}".format(cdf.df.shape))
dataset_df = cdf.df
# Show the first 5 rows of the DataFrame
print(dataset_df.head(5))