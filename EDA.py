import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
sns.set()
pd.set_option('display.max_column', 100)

dtypes={'session_id':'category', 
'elapsed_time':np.int32,
    'event_name':'category',
    'name':'category',
    'level':np.uint8,
    'page':'category',
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

data_train =  pd.read_csv("Data/train.csv")
data_train_label = pd.read_csv("Data/train_labels.csv")

df =  data_train.copy()
df_labels =  data_train_label.copy()

df = df.sort_values(['session_id','elapsed_time'])

print(df.head(10))

print((df.isna().sum()/df.shape[0]).sort_values(ascending=False))

# Not enough memory for this
#sns.heatmap(df.isna(), cbar=False)

















plt.show()