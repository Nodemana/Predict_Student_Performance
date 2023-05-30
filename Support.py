import tensorflow as tf
import tensorflow_addons as tfa


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# function to compare all predictors and return a collection of all predictors with a correlation of higher than a certain val (decimal)
def CorrelationMatrix(X_train, val):
    # create a correlation matrix
    corr_matrix = X_train.corr()
    corr_pairs = []

    # iterate through each element in the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # heck if the absval of the correlation is greater than 'val'
            if abs(corr_matrix.iloc[i, j]) > val:
                #add the pair to the list
                print(corr_matrix.columns[i] + " and " + corr_matrix.columns[j] + ": " + str(val))
                corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
    
    return corr_pairs

def RemoveCorrelatedColumns(X_train, corr_pairs):
    
    
    new_X_train = X_train.copy()
    
    cols_to_remove = []
    
    # iterate through each pair of correlated columns
    for pair in corr_pairs:
        #check if either column has already been marked for removal
        if pair[0] not in cols_to_remove and pair[1] not in cols_to_remove:
            #if neither column has been marked for removal, mark the second column for removal
            cols_to_remove.append(pair[1])
    
    # drop the columns marked for removal and return the  new_X_train
    new_X_train = new_X_train.drop(cols_to_remove, axis=1)
    return new_X_train

def Count_by_pval(results):
    #get the p-values from the fitted results
    p_values = results.pvalues
    
    #Count the number of predictors in each interval
    num_p_0_05 = np.sum((p_values >= 0) & (p_values <= 0.05))
    num_p_05_2 = np.sum((p_values > 0.05) & (p_values <= 0.2))
    num_p_2_5 = np.sum((p_values > 0.2) & (p_values <= 0.5))
    num_p_5 = np.sum(p_values > 0.5)
    
    #return as a dictionary
    return {'0 <= p <= 0.05': num_p_0_05,
            '0.05 < p <= 0.2': num_p_05_2,
            '0.2 < p <= 0.5': num_p_2_5,
            '0.5 < p': num_p_5}

def plot_Correlation_matrix(data):
    # Calculate the correlation matrix
    correlation_matrix = data.corr()

    # Create a heatmap plot of the correlation matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)

    
    plt.title('Correlation Matrix')
