import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.model_selection import KFold, GroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

from matplotlib import ticker
import time
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from sklearn.metrics import f1_score, ConfusionMatrixDisplay, classification_report, confusion_matrix

import gc

# Reduce Memory Usage
# reference : https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65 @ARJANGROEN

def reduce_memory_usage(df):
    
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype.name
        if ((col_type != 'datetime64[ns]') & (col_type != 'category')):
            if (col_type != 'object'):
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        pass
            else:
                df[col] = df[col].astype('category')
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage became: ",mem_usg," MB")
    
    return df

def summary(df):
    print(f'data shape: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values * 100
    summ['%missing'] = df.isnull().sum().values / len(df)
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['first value'] = df.loc[0].values
    summ['second value'] = df.loc[1].values
    summ['third value'] = df.loc[2].values
    
    return summ

# reference: https://www.kaggle.com/code/cdeotte/random-forest-baseline-0-664/notebook
def feature_engineer(train):
    dfs = []
    for c in count_var:
        tmp = train.groupby(['session_id','level_group'])[c].agg('nunique')
        tmp.name = tmp.name + '_nunique'
        dfs.append(tmp)
    for c in mean_var:
        tmp = train.groupby(['session_id','level_group'])[c].agg('mean')
        dfs.append(tmp)
    for c in event_var:
        tmp = train.groupby(['session_id','level_group'])[c].agg('sum')
        tmp.name = tmp.name + '_sum'
        dfs.append(tmp)
    df = pd.concat(dfs,axis=1)
    df = df.fillna(-1)
    df = df.reset_index()
    df = df.set_index('session_id')
    return df

train_df = pd.read_csv('Data/train.csv')
#train_df.info()

train_df = reduce_memory_usage(train_df)
#train_df.info()
gc.collect()

train_label = pd.read_csv('Data/train_labels.csv')
train_label = reduce_memory_usage(train_label)
train_label['session'] = train_label.session_id.apply(lambda x: int(x.split('_')[0]) )
train_label['q'] = train_label.session_id.apply(lambda x: int(x.split('_')[-1][1:]) )
#print( 'shape of label dataset is:',train_label.shape )
#print(train_label.head())
gc.collect()

summary_table = summary(train_df)
#print(summary_table)


#create dummies
just_dummies = pd.get_dummies(train_df['event_name'])

train_df = pd.concat([train_df, just_dummies], axis=1)
#print(train_df.head())

#print(train_df['event_name'].value_counts())

count_var = ['event_name', 'fqid','room_fqid', 'text']
mean_var = ['elapsed_time','level']
event_var = ['navigate_click','person_click','cutscene_click','object_click','map_hover','notification_click',
            'map_click','observation_click','checkpoint','elapsed_time']

df_tr = feature_engineer(train_df)
#print( df_tr.shape )
#print(df_tr.head())
gc.collect()


def plot_history(history):
    fig = plt.figure(figsize=[20, 6])
    ax = fig.add_subplot(1, 2, 1)
    plt.title("Loss vs Epochs", fontsize = 30)
    plt.xlabel("Epochs",fontsize = 20)
    plt.ylabel("Loss",fontsize = 20)
    ax.plot(history['loss'], label="Training Loss")
    ax.plot(history['val_loss'], label="Validation Loss")
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    plt.title("Accuracy vs Epochs", fontsize = 30)
    plt.xlabel("Epochs",fontsize = 20)
    plt.ylabel("Accuracy",fontsize = 20)
    ax.plot(history['accuracy'], label="Training Accuracy")
    ax.plot(history['val_accuracy'], label="Validation Accuracy")
    ax.legend()

# DEFINE THE NEURAL NETWORK MODEL UP TO THE EMBEDDING LAYER
def create_embedding_model(input_dim):
    model = Sequential()
    model.add(Dense(1024, input_dim=input_dim, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.002), metrics=['accuracy'])
    return model



from sklearn.metrics import accuracy_score, roc_auc_score


def evaluate_model(model, X_test, y_test, threshold):
    # Compute prediction probabilities
    y_pred_prob = model.predict(X_test).flatten()

    # Compute predicted classes
    y_pred = (y_pred_prob > threshold).astype(int)

    # Compute accuracy
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy:', acc)

    # Compute ROC AUC score
    auc = roc_auc_score(y_test, y_pred_prob)
    print('ROC AUC score:', auc)

    # Print confusion matrix
    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 2, 1)
    conf_matrix = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(conf_matrix).plot(ax=ax)
    ax.set_title('Training Set Performance: %s' % (sum(y_pred == y_test)/len(y_test)))

    # Print classification report
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
#check data type
#print(df_tr.dtypes)

FEATURES = [c for c in df_tr.columns if c != 'level_group']
print('We will train with', len(FEATURES) ,'features')
ALL_USERS = df_tr.index.unique()
print('We will train with', len(ALL_USERS) ,'users info')

# Initialise a cross validation framework
gkf = GroupKFold(n_splits=5)
oof = pd.DataFrame(data=np.zeros((len(ALL_USERS),18)), index=ALL_USERS)
print(oof.head())
models = {}

# COMPUTE CV SCORE WITH 5 GROUP K FOLD
# 
for i, (train_index, test_index) in enumerate(gkf.split(X=df_tr, groups=df_tr.index)):
    print('#'*25)
    print('### Fold',i+1)
    print('#'*25)
    
    xgb_params = {
    'objective' : 'binary:logistic',
    'eval_metric':'logloss',
    'learning_rate': 0.05,
    'max_depth': 4,
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
    'tree_method':'hist',
    'subsample':0.8,
    'colsample_bytree': 0.4,
    'use_label_encoder' : False}
    
    # ITERATE THRU QUESTIONS 1 THRU 18
    for t in range(1,19):
        print(t,', ',end='')
        
        # USE THIS TRAIN DATA WITH THESE QUESTIONS
        if t<=3: grp = '0-4'
        elif t<=13: grp = '5-12'
        elif t<=22: grp = '13-22'
            
        # TRAIN DATA
        train_x = df_tr.iloc[train_index]
        train_x = train_x.loc[train_x.level_group == grp]
        train_users = train_x.index.values
        train_y = train_label.loc[train_label.q==t].set_index('session').loc[train_users]
        
        # VALID DATA
        valid_x = df_tr.iloc[test_index]
        valid_x = valid_x.loc[valid_x.level_group == grp]
        valid_users = valid_x.index.values
        valid_y = train_label.loc[train_label.q==t].set_index('session').loc[valid_users]

        # SCALE THE DATA
        scaler = StandardScaler()
        train_x_scaled = scaler.fit_transform(train_x[FEATURES].astype('float32'))
        valid_x_scaled = scaler.transform(valid_x[FEATURES].astype('float32'))
        
        # TRAIN MODEL
        # create the embedding model and load weights from the previously trained model
        embedding_model = create_embedding_model(train_x_scaled.shape[1])
        embedding_model.load_weights('Trained_Weights/Model2')
        #early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        #history = embedding_model.fit(train_x_scaled, train_y['correct'], validation_data=(valid_x_scaled, valid_y['correct']), epochs=100, callbacks=[early_stopping])

        #plot_history(history.history)
        aux_train_x = embedding_model.predict(train_x_scaled)
        aux_valid_x = embedding_model.predict(valid_x_scaled)
        # ADD Auxilary classification head here:
         # TRAIN MODEL        
        clf =  XGBClassifier(**xgb_params)
        clf.fit(aux_train_x.astype('float32'), train_y['correct'],
                eval_set=[ (aux_valid_x.astype('float32'), valid_y['correct']) ],
                verbose=0)
        print(f'{t}({clf.best_ntree_limit}), ',end='')
        
        # SAVE MODEL, PREDICT VALID OOF
        models[f'{grp}_{t}'] = clf
        oof.loc[valid_users, t-1] = clf.predict_proba(aux_valid_x.astype('float32'))[:,1]



#Saves the trained weights
#embedding_model.save_weights('Trained_Weights/Model2')
#PUT TRUE LABELS INTO DATAFRAME WITH 18 COLUMNS
true = oof.copy()
for k in range(18):
    # GET TRUE LABELS
    tmp = train_label.loc[train_label.q == k+1].set_index('session').loc[ALL_USERS]
    true[k] = tmp.correct.values

# FIND BEST THRESHOLD TO CONVERT PROBS INTO 1s AND 0s
scores = []; thresholds = []
best_score = 0; best_threshold = 0

for threshold in np.arange(0.4,0.81,0.01):
    print(f'{threshold:.02f}, ',end='')
    preds = (oof.values.reshape((-1))>threshold).astype('int')
    m = f1_score(true.values.reshape((-1)), preds, average='macro')   
    scores.append(m)
    thresholds.append(threshold)
    if m>best_score:
        best_score = m
        best_threshold = threshold

print('When using optimal threshold...')
for k in range(18):
        
    # COMPUTE F1 SCORE PER QUESTION
    m = f1_score(true[k].values, (oof[k].values>best_threshold).astype('int'), average='macro')
    print(f'Q{k}: F1 =',m)
    
# COMPUTE F1 SCORE OVERALL
m = f1_score(true.values.reshape((-1)), (oof.values.reshape((-1))>best_threshold).astype('int'), average='macro')
print('==> Overall F1 =',m)

# Get the model that corresponds to the best threshold
best_model_key = f'{grp}_{t}'  # substitute with appropriate keys

# Ensure y_test is an array of integers
valid_y_int = np.array(valid_y['correct']).astype(int)

# Compute prediction probabilities using best model
y_pred_prob_best = models[best_model_key].predict(embedding_model.predict(valid_x_scaled)).flatten()

# Call the evaluation function
evaluate_model(models[best_model_key], embedding_model.predict(valid_x_scaled), valid_y_int, best_threshold)


# PLOT THRESHOLD VS. F1_SCORE
plt.figure(figsize=(20,5))
plt.plot(thresholds,scores,'-o',color='blue')
plt.scatter([best_threshold], [best_score], color='blue', s=300, alpha=1)
plt.xlabel('Threshold',size=14)
plt.ylabel('Validation F1 Score',size=14)
plt.title(f'Threshold vs. F1_Score with Best F1_Score = {best_score:.3f} at Best Threshold = {best_threshold:.3}',size=18)
plt.show()