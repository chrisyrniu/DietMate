from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import sensor_data
import pandas as pd
import plot

from scipy.stats import mode 
from sklearn import metrics


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):

    args = parser.parse_args(argv[1:])
    
    # Fetch the data.
    (cv_x, cv_y, cv_x_df) = sensor_data.load_data()
    cv_y.index = range(0,cv_y.shape[0])


    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in cv_x_df.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=5)

    # Cross validation
    n_splits = 50
    dataset = sensor_data.cv_dataset(cv_x, cv_y, n_splits)
    accu = 0
    step = 0
    pred_label = []
    print(type(cv_y))

    for x_train, y_train, x_test, y_test in dataset:
        x_train = pd.DataFrame(x_train, columns = ['Mean_Piezo', 'Std_Piezo', 'Mean_Strain', 'Std_Strain', 'Mean_Mic', 'Std_Mic'])
        y_train = pd.DataFrame(y_train)
        x_test = pd.DataFrame(x_test, columns = ['Mean_Piezo', 'Std_Piezo', 'Mean_Strain', 'Std_Strain', 'Mean_Mic', 'Std_Mic'])
        y_test = pd.DataFrame(y_test)

        step = step + 1

        classifier.train(
            input_fn=lambda:sensor_data.train_input_fn(x_train, y_train,
                                                     args.batch_size),
            steps=args.train_steps)
        
        predictions = classifier.predict(
            input_fn=lambda:sensor_data.eval_input_fn(x_test,
                                                labels=None,
                                                batch_size=args.batch_size))

        for pred_dict in predictions:
            pred_label.append(pred_dict['class_ids'][0])


    pred_label = pd.Series(pred_label)

    # Apply mode filter to the estimated labels.
    filter_mode = lambda x: mode(x)[0][0]
    window_filter = 51
    pred_label = pred_label.rolling(window=window_filter, center=True).apply(filter_mode, raw=True)

    cv_y.index = range(0,cv_y.shape[0])
    pred_label.index = range(0,pred_label.shape[0])

    # Eliminate the begining and the end of the labels after the mode filtering.
    for i in range(0,cv_y.shape[0]):
        if i<=((window_filter-1)/2-1) or i>=(cv_y.shape[0]-(window_filter-1)/2):
            cv_y = cv_y.drop([i])
            pred_label = pred_label.drop([i])

    cv_y.index = range(0,cv_y.shape[0])
    pred_label.index = range(0,pred_label.shape[0])

    precision = metrics.precision_score(cv_y, pred_label, average = 'weighted')
    f_measure = metrics.f1_score(cv_y, pred_label, average = 'weighted')
    recall = metrics.recall_score(cv_y, pred_label, average = 'weighted')

    # Calculate and output the accuracy.
    print('\nTest set accuracy: %f \n'  % (1-(((cv_y != pred_label).sum())/(cv_y.shape[0]))))

    # Output the precision, recall and f-measure scores.f-measure.
    print('\nPrecision score: %f \n'  % precision)
    print('\nRecall score: %f \n'  % recall)
    print('\nF-measure score: %f \n'  % f_measure)

    cv_x = sensor_data.sensorData()
    cv_x_piezo = cv_x[0]
    cv_x_strain = cv_x[1]
    cv_x_mic = cv_x[2]

    max_window_size = sensor_data.transferWindowSize()

    # Prune the original sensor data.
    for i in range(0,cv_x_piezo.shape[0]):
        if i<=((max_window_size + window_filter - 2)/2 - 1) or i>=(cv_x_piezo.shape[0]-(max_window_size + window_filter - 2)/2):
            cv_x_piezo = cv_x_piezo.drop([i])
            cv_x_strain = cv_x_strain.drop([i])
            cv_x_mic = cv_x_mic.drop([i])

    cv_x_piezo.index = range(0,cv_x_piezo.shape[0])
    cv_x_strain.index = range(0,cv_x_strain.shape[0])
    cv_x_mic.index = range(0,cv_x_mic.shape[0])
    
    # Plot the results.
    plot.draw_coloredLabel(cv_y, pred_label, (-500, cv_y.shape[0]+5000))
    plot.draw_sensorData(cv_x_piezo, cv_y, pred_label, (-500, cv_y.shape[0]+4000))
    plot.draw_sensorData(cv_x_strain, cv_y, pred_label, (-500, cv_y.shape[0]+4000))
    plot.draw_sensorData(cv_x_mic, cv_y, pred_label, (-500, cv_y.shape[0]+4000))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
