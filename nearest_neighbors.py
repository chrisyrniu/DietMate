import pandas as pd
import numpy as np
import csv
import plot

from scipy.stats import mode 
from sklearn import neighbors 

from sklearn.model_selection import cross_val_predict
from sklearn import metrics

CSV_COLUMN_NAMES = ['Piezo', 'Strain', 'Mic', 'Activities']
ACTIVITIES = ['Talking', 'Opening', 'Chewing', 'Swallowing', 'Other']

# 'cv' = 'cross validation'

raw_cv_path = "C:/Users/dell/Desktop/AICPS/health_iot_project/Chris/8. multi_5cls/data/5subjects.csv"
inter_cv_path = "C:/Users/dell/Desktop/AICPS/health_iot_project/Chris/8. multi_5cls/data/cv.csv"

with open(raw_cv_path, 'rt', newline='') as inp, open(inter_cv_path, 'wt', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        writer.writerow([row[1], row[5], row[6], row[7]])

cv = pd.read_csv(inter_cv_path, names=CSV_COLUMN_NAMES, header=None)
cv_x, cv_y = cv, cv.pop('Activities')

cv_x_rep = cv_x
cv_x_piezo = cv_x['Piezo']
cv_x_strain = cv_x['Strain']
cv_x_mic = cv_x['Mic']

max_score = 0

for window_size1 in range(33, 34 , 2):
    for window_size2 in range(33, 34, 2):
        for window_size3 in range(33, 34, 2):
            for window_size4 in range(33, 34, 2):
                for window_size5 in range(33, 34, 2):
                    for window_size6 in range(33, 34, 2):
                        for window_filter in range(51, 52, 2):

                            window_size = max(window_size1, window_size2, window_size3, window_size4, window_size5, window_size6)

                            cv = pd.read_csv(inter_cv_path, names=CSV_COLUMN_NAMES, header=None)
                            cv_x, cv_y = cv, cv.pop('Activities')

                            cv_x_mean_piezo = cv_x['Piezo'].rolling(window=window_size1, center=True).mean()
                            cv_x_std_piezo = cv_x['Piezo'].rolling(window=window_size2, center=True).std()
                            cv_x_mean_strain = cv_x['Strain'].rolling(window=window_size3, center=True).mean()
                            cv_x_std_strain = cv_x['Strain'].rolling(window=window_size4, center=True).std()
                            cv_x_mean_mic = cv_x['Mic'].rolling(window=window_size5, center=True).mean()
                            cv_x_std_mic = cv_x['Mic'].rolling(window=window_size6, center=True).std()

                            for i in range(0,cv_x.shape[0]):
                                if i<=((window_size-1)/2-1) or i>=(cv_x.shape[0]-(window_size-1)/2):
                                    cv_x_mean_piezo = cv_x_mean_piezo.drop([i])
                                    cv_x_std_piezo = cv_x_std_piezo.drop([i])
                                    cv_x_mean_strain = cv_x_mean_strain.drop([i])
                                    cv_x_std_strain = cv_x_std_strain.drop([i])
                                    cv_x_mean_mic = cv_x_mean_mic.drop([i])
                                    cv_x_std_mic = cv_x_std_mic.drop([i])
                                    cv_y = cv_y.drop([i])

                            cv_x = np.array([cv_x_mean_piezo, cv_x_std_piezo, cv_x_mean_strain, cv_x_std_strain, cv_x_mean_mic, cv_x_std_mic])
                            cv_x = np.transpose(cv_x)

                            k_neighbors = 15
                            weights =  'uniform'
                            knn = neighbors.KNeighborsClassifier(k_neighbors, weights)
                            predicted = cross_val_predict(knn, cv_x, cv_y, cv = 50)
                            predicted = pd.Series(predicted)

                            filter_mode = lambda x: mode(x)[0][0]
                            predicted = predicted.rolling(window=window_filter, center=True).apply(filter_mode, raw=True)

                            cv_y.index = range(0,cv_y.shape[0])
                            predicted.index = range(0,predicted.shape[0])
                            cv_y_rep = cv_y

                            for i in range(0,cv_y.shape[0]):
                                if i<=((window_filter-1)/2-1) or i>=(cv_y_rep.shape[0]-(window_filter-1)/2):
                                    cv_y = cv_y.drop([i])
                                    predicted = predicted.drop([i])

                            cv_y.index = range(0,cv_y.shape[0])
                            predicted.index = range(0,predicted.shape[0])

                            if metrics.accuracy_score(cv_y, predicted) > max_score:
                               max_score = metrics.accuracy_score(cv_y, predicted)
                               max_win1 = window_size1
                               max_win2 = window_size2
                               max_win3 = window_size3
                               max_win4 = window_size4
                               max_win5 = window_size5
                               max_win6 = window_size6
                               max_win_filter = window_filter
                               cv_y_mark = cv_y
                               predicted_mark = predicted
                               # precision = metrics.precision_score(cv_y, predicted, average = 'micro')
                               f_measure = metrics.f1_score(cv_y, predicted, average = 'micro')
                               recall = metrics.recall_score(cv_y, predicted, average = 'micro')



print('\nTest set accuracy: %f \n'  % max_score)
# print('\nPrecision score: %f \n'  % precision)
print('\nF-measure score: %f \n'  % f_measure)
print('\nRecall score: %f \n'  % recall)
print('Window size 1: %f \n'  % max_win1)
print('Window size 2: %f \n'  % max_win2)
print('Window size 3: %f \n'  % max_win3)
print('Window size 4: %f \n'  % max_win4)
print('Window size 5: %f \n'  % max_win5)
print('Window size 6: %f \n'  % max_win6)
print('Window size filter: %f \n'  % max_win_filter)

max_win = max(max_win1, max_win2, max_win3, max_win4, max_win5, max_win6)

for i in range(0,cv_x_rep.shape[0]):
    if i<=((max_win + max_win_filter - 2)/2 - 1) or i>=(cv_x_rep.shape[0]-(max_win + max_win_filter - 2)/2):
        cv_x_piezo = cv_x_piezo.drop([i])
        cv_x_strain = cv_x_strain.drop([i])
        cv_x_mic = cv_x_mic.drop([i])

cv_x_piezo.index = range(0,cv_x_piezo.shape[0])
cv_x_strain.index = range(0,cv_x_strain.shape[0])
cv_x_mic.index = range(0,cv_x_mic.shape[0])

plot.draw_coloredLabel(cv_y_mark, predicted_mark, (-500, cv_y_mark.shape[0]+5000))
plot.draw_label(cv_y_mark, predicted_mark, (-500, cv_y_mark.shape[0]+500))
plot.draw_sensorData(cv_x_piezo, cv_y_mark, predicted_mark, (-500, cv_x_piezo.shape[0]+4000))
plot.draw_sensorData(cv_x_strain, cv_y_mark, predicted_mark, (-500, cv_x_piezo.shape[0]+4000))
plot.draw_sensorData(cv_x_mic, cv_y_mark, predicted_mark, (-500, cv_x_piezo.shape[0]+4000))
