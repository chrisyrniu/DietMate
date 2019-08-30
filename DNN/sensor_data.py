import pandas as pd
import tensorflow as tf
import numpy as np
import csv

from sklearn.model_selection import LeavePOut, KFold

CSV_COLUMN_NAMES = ['Piezo', 'Strain', 'Mic', 'Activities']
ACTIVITIES = ['Talking', 'Opening', 'Chewing', 'Swallowing', 'Other']

# 'cv' = 'cross validation'

raw_cv_path = "../data/5subjects.csv"
inter_cv_path = "../data/cv.csv"

with open(raw_cv_path, 'rt', newline='') as inp, open(inter_cv_path, 'wt', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        writer.writerow([row[1], row[5], row[6], row[7]])

    # Fix the window sizes
    window_size1 = 33 
    window_size2 = 33
    window_size3 = 33  
    window_size4 = 33 
    window_size5 = 33
    window_size6 = 33  

    window_size = max(window_size1, window_size2, window_size3, window_size4, window_size5, window_size6)

def load_data(y_name='Activities'):

    cv = pd.read_csv(inter_cv_path, names=CSV_COLUMN_NAMES, header=None)
    cv_x, cv_y = cv, cv.pop('Activities')

    # Feature extraction, using the sliding window.
    cv_x_mean_piezo = cv_x['Piezo'].rolling(window=window_size1, center=True).mean()
    cv_x_std_piezo = cv_x['Piezo'].rolling(window=window_size2, center=True).std()
    cv_x_mean_strain = cv_x['Strain'].rolling(window=window_size3, center=True).mean()
    cv_x_std_strain = cv_x['Strain'].rolling(window=window_size4, center=True).std()
    cv_x_mean_mic = cv_x['Mic'].rolling(window=window_size5, center=True).mean()
    cv_x_std_mic = cv_x['Mic'].rolling(window=window_size6, center=True).std()

    # After using the sliding window, eliminate the begining and the end of the dataset.
    for i in range(0,cv_x.shape[0]):
        if i<=((window_size-1)/2-1) or i>=(cv_x.shape[0]-(window_size-1)/2):
            cv_x_mean_piezo = cv_x_mean_piezo.drop([i])
            cv_x_std_piezo = cv_x_std_piezo.drop([i])
            cv_x_mean_strain = cv_x_mean_strain.drop([i])
            cv_x_std_strain = cv_x_std_strain.drop([i])
            cv_x_mean_mic = cv_x_mean_mic.drop([i])
            cv_x_std_mic = cv_x_std_mic.drop([i])
            cv_y = cv_y.drop([i])

    cv_x_df = pd.DataFrame({'Mean_Piezo': cv_x_mean_piezo, 'Std_Piezo': cv_x_std_piezo, 'Mean_Strain': cv_x_mean_strain, 'Std_Strain': cv_x_std_strain, 'Mean_Mic': cv_x_mean_mic, 'Std_Mic': cv_x_std_mic})
    cv_x = np.array([cv_x_mean_piezo, cv_x_std_piezo, cv_x_mean_strain, cv_x_std_strain, cv_x_mean_mic, cv_x_std_mic])
    cv_x = np.transpose(cv_x)
    return (cv_x, cv_y, cv_x_df)

def cv_dataset(x_data, y_data, n_splits):
    for train_index, test_index in KFold(n_splits).split(x_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        yield x_train, y_train, x_test, y_test
    

def train_input_fn(features, labels, batch_size):

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(10000).repeat().batch(batch_size)

    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

def sensorData():
    cv = pd.read_csv(inter_cv_path, names=CSV_COLUMN_NAMES, header=None)
    cv_x, cv_y = cv, cv.pop('Activities')
    cv_x_piezo = cv_x['Piezo']
    cv_x_strain = cv_x['Strain']
    cv_x_mic = cv_x['Mic']

    return [cv_x_piezo, cv_x_strain, cv_x_mic]

def transferWindowSize():
    return max(window_size1, window_size2, window_size3, window_size4, window_size5, window_size6)


