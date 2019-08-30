import tensorflow as tf
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

from sklearn import naive_bayes 
from sklearn.externals import joblib

CSV_COLUMN_NAMES = ['Piezo', 'Strain', 'Activities']
ACTIVITIES = ['Other', 'Chewing', 'Swallowing' ]

# train_path = "C:/Users/dell/Desktop/AICPS/health_iot_project/data/exp/train_data_nuts.csv"
# test_path = "C:/Users/dell/Desktop/AICPS/health_iot_project/data/exp/test_data_nuts.csv"

raw_train_path = "../data/train_data_nuts.csv"
raw_test_path = "../data/test_data_nuts.csv"
inter_train_path = "../data/train_piezo.csv"
inter_test_path = "../data/test_piezo.csv"

with open(raw_train_path, 'rt', newline='') as inp, open(inter_train_path, 'wt', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[7] == " 0" or row[7] ==" 1" or row[7] ==" 2" or row[7] ==" 3" :
           row[7] = " 0"
        if row[7] == " 4" :
           row[7] = " 1"
        if row[7] == " 5" :
           row[7] = " 2"
        writer.writerow([row[1], row[5], row[7]])

with open(raw_test_path, 'rt', newline='') as inp, open(inter_test_path, 'wt', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[7] == " 0" or row[7] ==" 1" or row[7] ==" 2" or row[7] ==" 3" :
           row[7] = " 0"
        if row[7] == " 4" :
           row[7] = " 1"
        if row[7] == " 5" :
           row[7] = " 2"
        writer.writerow([row[1], row[5], row[7]])

train = pd.read_csv(inter_train_path, names=CSV_COLUMN_NAMES, header=None)
train_x, train_y = train, train.pop('Activities')

test = pd.read_csv(inter_test_path, names=CSV_COLUMN_NAMES, header=None)
test_x, test_y = test, test.pop('Activities')


# Mean and standard variation value of Piezo
window_size1 = 11
window_size2 = 5
window_size3 = 3
window_size4 = 7
window_size = max(window_size1, window_size2, window_size3, window_size4)

train_x_piezo_mean = train_x['Piezo'].rolling(window=window_size1, center=True).mean()
train_x_piezo_std = train_x['Piezo'].rolling(window=window_size2, center=True).std()
train_x_strain_mean = train_x['Strain'].rolling(window=window_size3, center=True).mean()
train_x_strain_std = train_x['Strain'].rolling(window=window_size4, center=True).std()

for i in range(0,train_x.shape[0]):
    if i<=((window_size-1)/2-1) or i>=(train_x.shape[0]-(window_size-1)/2):
        train_x_piezo_mean = train_x_piezo_mean.drop([i])
        train_x_piezo_std = train_x_piezo_std.drop([i])
        train_x_strain_mean = train_x_strain_mean.drop([i])
        train_x_strain_std = train_x_strain_std.drop([i])
        train_y = train_y.drop([i])

test_x_piezo_mean = test_x['Piezo'].rolling(window=window_size1, center=True).mean()
test_x_piezo_std = test_x['Piezo'].rolling(window=window_size2, center=True).std()
test_x_strain_mean = test_x['Strain'].rolling(window=window_size3, center=True).mean()
test_x_strain_std = test_x['Strain'].rolling(window=window_size4, center=True).std()

for i in range(0,test_x.shape[0]):
    if i<=((window_size-1)/2-1) or i>=(test_x.shape[0]-(window_size-1)/2):
        test_x_piezo_mean = test_x_piezo_mean.drop([i])
        test_x_piezo_std = test_x_piezo_std.drop([i])
        test_x_strain_mean = test_x_strain_mean.drop([i])
        test_x_strain_std = test_x_strain_std.drop([i])
        test_y = test_y.drop([i])

train_x = np.array([train_x_piezo_mean, train_x_piezo_std, train_x_strain_mean, train_x_strain_std])
train_x = np.transpose(train_x)

test_x = np.array([test_x_piezo_mean, test_x_piezo_std, test_x_strain_mean, test_x_strain_std])
test_x = np.transpose(test_x)

gnb = naive_bayes.GaussianNB()
y_pred = gnb.fit(train_x, train_y)

# Save the model.
joblib.dump(y_pred, "naive_bayes.m")
# Reload the model.
y_pred_reload = joblib.load("naive_bayes.m")
print('\nTest set accuracy: %f \n'  % (y_pred_reload.score(test_x, test_y)))
