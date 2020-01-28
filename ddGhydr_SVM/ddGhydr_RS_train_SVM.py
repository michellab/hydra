# General:
import pandas as pd
import numpy as np
import os
import csv
import time
import shutil
import logging
import pickle
import sys

# SVM:
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# SciKit-Optimise:
from skopt import gp_minimize, dump
from skopt.space import Categorical
from skopt.utils import use_named_args

# Path variables:
path = './'
datasets_dr = '../datasets/'
SDF_dr = datasets_dr + 'sdffiles/'
output_dr = path + 'output/'
if not os.path.exists(output_dr):
    os.mkdir(output_dr)
figures_dr = path + 'figures/'
if not os.path.exists(figures_dr):
    os.mkdir(figures_dr)

# Global variables:
model_type = 'SVM'
label_col = 'dGoffset (kcal/mol)'

# DNN global variables
n_calls = 40  # Number of Bayesian optimisation loops for hyperparameter optimisation, 40 is best for convergence, > 60 scales to very expensive
best_mae = np.inf

# load in data set
dtrain_df = pd.read_hdf(datasets_dr + 'dtrain_data.h5', key='relative')
num_input_nodes = len(dtrain_df.columns) - 1


def main():

    # initiate log file
    logging.basicConfig(filename= output_dr + 'training_logfile.txt',
                    filemode='a',
                    format='%(asctime)s - %(message)s',
                    level=logging.INFO)
    logging.info('Starting {}.'.format(__file__))

    train_model(dtrain_df)

    logging.info('Finished {}.'.format(__file__))


def create_model(param_gamma, param_C, param_epsilon):
    """Returns a SVR class instance."""

    return SVR(gamma=param_gamma, C=param_C, epsilon=param_epsilon, verbose=True)


def train_model(train_set):
    """
    1. Unpack training data.
    2. Define hyper-perameter ranges.
    3. Define early stopping perameters.
    4. Optimise hyper-perameters and save best model.
    5. Save mae per call to CSV.
    """

    # seperate features and labels and convert to numpy array
    X = train_set.drop(label_col, axis=1).to_numpy()
    y = train_set.pop(label_col).to_numpy()

    # retrieve data sets
    train_X, validate_X, train_y, validate_y = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # init recording statistics
    with open(output_dr + model_type + '_statistics.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['MAE (kcal/mol)', 'parameters'])

    # set hyper-parameter ranges, append to list
    dim_param_C = Categorical(categories=list(np.logspace(-3, 2, 6, dtype="float32")), name="param_C")
    dim_param_gamma = Categorical(categories=list(np.logspace(-3, 2, 6, dtype="float32")), name="param_gamma")
    dim_param_epsilon = Categorical(categories=list(np.logspace(-3, 2, 6, dtype="float32")), name="param_epsilon")

    # gp_minimize dimensions
    dimensions = [dim_param_C, dim_param_gamma, dim_param_epsilon]

    # start hyper-perameter optimisation
    @use_named_args(dimensions=dimensions)
    def fitness(param_C, param_gamma, param_epsilon):
        """Function for gaussian process optmisation."""

        # create SVR model
        model = create_model(param_C, param_gamma, param_epsilon)

        # train model on training data
        model.fit(train_X, train_y)

        # validate model
        predicted_y = model.predict(validate_X)
        mae = mean_absolute_error(validate_y, predicted_y)

        # update statistics
        with open(output_dr + model_type + '_statistics.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow([mae, [param_gamma, param_gamma, param_epsilon]])

        # check if model improves
        global best_mae
        if mae < best_mae:
            # update new model accuracy.
            best_mae = mae
            # overwrite model if mae improves
            pkl_file = output_dr + 'ddGhydr_' + model_type + '_model.pickle'
            with open(pkl_file, 'wb') as file: pickle.dump(model, file)
            logging.info('Saved {}.'.format(pkl_file))

        return mae

    # a place for optimiser to start looking
    default_parameters = [1.0, 1.0, 1.0]

    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI',  # Expected Improvement.
                                n_calls=n_calls,
                                x0=default_parameters,
                                verbose=True)

    # save skopt object and analyse in a separate script as
    # https://github.com/scikit-optimize/scikit-optimize/blob/master/examples/bayesian-optimization.ipynb
    # https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/19_Hyper-Parameters.ipynb
    dump(search_result, output_dr + 'gp_minimize_result.pickle', store_objective=False)
    logging.info('Saved {}gp_minimize_result.pickle.'.format(output_dr))

    logging.info('Final parameters: {}.'.format(search_result.x))
    return search_result


# https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


if __name__ == '__main__':

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Script started on {}'.format(time.ctime()))

    # prevent accidental clearance of previous models in output/
    question = 'Are you sure you want to clear saved models in ./output/ ??'
    query = query_yes_no(question, 'no')
    if not query:
        print('Script terminated.')
        sys.exit()
    elif query:
        print('./output/ cleared.')

    # clean slate output_dr
    if os.path.exists(output_dr):
        shutil.rmtree(output_dr)
    if not os.path.exists(output_dr):
        os.mkdir(output_dr)

    main()

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Script finished.')
