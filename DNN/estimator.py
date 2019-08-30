from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import sensor_data
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):

    args = parser.parse_args(argv[1:])

    (cv_x, cv_y, cv_x_df) = sensor_data.load_data()
    cv_y.index = range(0,cv_y.shape[0])

    my_feature_columns = []
    for key in cv_x_df.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=5)
    
    n_splits = 50
    dataset = sensor_data.cv_dataset(cv_x, cv_y, n_splits)
    accu = 0
    step = 0

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
        eval_result = classifier.evaluate(
            input_fn=lambda:sensor_data.eval_input_fn(x_test, y_test,
                                                    args.batch_size))
        
        accu = accu + (eval_result["accuracy"])
        print('\nTest set accuracy: %f \n' % (accu/step))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
