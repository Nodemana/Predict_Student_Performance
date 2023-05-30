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
import tensorflow_decision_forests as tfdf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#show versions of the APIs
print("TensorFlow Decision Forests v" + tfdf.__version__)
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




dataset_df = pd.read_csv('/Data/train.csv', dtype=dtypes)
print("Full train dataset shape is {}".format(dataset_df.shape))

dataset_df.head(5)