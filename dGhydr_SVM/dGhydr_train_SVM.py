# General:
import pandas as pd
import numpy as np
import os
import csv
import time
import shutil
import logging
import pickle

# SciKit-Optimise:
from skopt import gp_minimize, dump
from skopt.space import Categorical
from skopt.utils import use_named_args

# SVM:
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

# Path variables:
path = './'
datasets_dr = '../datasets/'
SDF_dr = datasets_dr + 'sdffiles/'
output_dr = path + 'output/'

# Global variables:
model_type = 'SVM'
offset_col_name = 'dGoffset (kcal/mol)'

# set data processing configurations:
n_calls = 40  # Number of Bayesian optimisation loops for hyperparameter optimisation, 40 is best for convergence, > 60 scales to very expensive
best_mae = np.inf  # Point to consider top-performing model from (MAE/MAD); 1.0 = no improvement on test-set variance

# KFold parameters:
n_splits = 5  # Number of K-fold splits
random_state = 2  # Random number seed


def main():

    # initiate log file
    logging.basicConfig(filename= output_dr + 'train_logfile.txt',
                        filemode='a',
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)

    logging.info('Starting {}.'.format(__file__))
    logging.info('\n\nParameters:\n\nn_calls = {}  # gp_minimize\nn_splits = {}  # Kfolds\n'.format(n_calls, n_splits))


    # Load in dataset.
    train_df = pd.read_hdf(datasets_dr + 'train_data.h5', key='absolute')

    # training
    kfolds = split_dataset(train_df, n_splits, random_state)

    for i, fold in enumerate(kfolds):
        fold_num = i + 1
        train_model(fold, fold_num)

    logging.info('Finished dGhydr_{}.py.'.format(model_type))


def train_model(fold, fold_num):
    """
    1. Unpack fold into training, validating x and Y
    2. Define SVR starting hyperparameters
    3. Setup SVR classifier
    4. Determine statistics for validating-y against SVR-predicted-y
    5. Pickle clasifier

    Returns:
    1. Skopt object
    2. Best performing model
    """
    logging.info('Started training fold {}...'.format(str(fold_num)))

    # retrieve data sets and convert to numpy array
    train_X = fold[0][0].values
    validate_X = fold[0][1].values
    train_y = fold[1][0].values
    validate_y = fold[1][1].values

    # init recording statistics
    with open(output_dr + model_type + '_statistics.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['fold', 'MAE (kcal/mol)', 'parameters'])

    # set hyper-parameter ranges, append to list
    dim_param_C = Categorical(categories=list(np.logspace(-3, 2, 6, dtype="float32")), name="param_C")
    dim_param_gamma = Categorical(categories=list(np.logspace(-3, 2, 6, dtype="float32")), name="param_gamma")
    dim_param_epsilon = Categorical(categories=list(np.logspace(-3, 2, 6, dtype="float32")), name="param_epsilon")

    dimensions = [dim_param_C, dim_param_gamma, dim_param_epsilon]

    # start optimising hyper-parameters
    ############ add logging decorator ############
    @use_named_args(dimensions=dimensions)
    def fitness(param_C, param_gamma, param_epsilon):

        # create SVR instance
        model = SVR(gamma=param_gamma, C=param_C,
                    epsilon=param_epsilon, verbose=False)

        # train model on training data
        model.fit(train_X, train_y)

        # validate model
        predicted_y = model.predict(validate_X)

        # calculate some statistics on validate set
        # note: different args orders
        mae = mean_absolute_error(validate_y, predicted_y)
        # r2 = model.score(predicted_y, validate_y)

        # update statistics
        with open(output_dr + model_type + '_statistics.csv', 'a') as file:
            writer = csv.writer(file)
            # writer.writerow([mae, r2, [param_gamma, param_gamma, param_epsilon]])
            writer.writerow([fold_num, mae, [param_gamma, param_gamma, param_epsilon]])

        # print('\nMAE = {} kcal/mol\nr2 = {}\n'.format(mae, r2))
        print('MAE = {} kcal/mol'.format(mae))
        print('Parameters: {}\n'.format([param_gamma, param_gamma, param_epsilon]))

        global best_mae
        if mae < best_mae:
            # save model
            with open(output_dr + 'fold_' + str(fold_num) + '_' + model_type + '_model.pickle', 'wb') as file:
                pickle.dump(model, file)
            logging.info('Model saved at ' + output_dr + 'fold_' + str(fold_num) + '_' + model_type + '_model.pickle')

            # Update the regressor accuracy.
            best_mae = mae

        return mae

    default_parameters = [1.0, 1.0, 1.0]

    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI',  # Expected Improvement.
                                n_calls=n_calls,
                                x0=default_parameters,
                                verbose=False)

    print('Concluded optimal hyper-parameters:')
    print('Fold {}: {}'.format(str(fold_num), search_result.x))

    # save skopt object and analyse in a separate script as
    dump(search_result, output_dr + 'fold_' + str(fold_num) + '_gp_minimize_result.pickle', store_objective=False)
    logging.info('Saved {}fold_{}_gp_minimize_result.pickle.'.format(output_dr, fold_num))

    logging.info('Finished training fold {}: {}.'.format(str(fold_num), search_result.x))
    return search_result


def split_dataset(dataset, n_splits, random_state):
    """KFold implementation for pandas DataFrame.
    (https://stackoverflow.com/questions/45115964/separate-pandas-dataframe-using-sklearns-kfold)"""
    logging.info('Performing {}-fold cross-validation...'.format(n_splits))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    kfolds = []
    global offset_col_name

    for train, validate in kf.split(dataset):
        training = dataset.iloc[train]
        train_labels = training[offset_col_name]
        train_set = training.drop(offset_col_name, axis=1)

        validating = dataset.iloc[validate]
        validate_labels = validating[offset_col_name]
        validate_set = validating.drop(offset_col_name, axis=1)

        kfolds.append(
            [[train_set, validate_set],
             [train_labels, validate_labels]]
        )

    with open(path + 'kfolds.json', "wb") as f:
        pickle.dump(kfolds, f)

    logging.info('Pickled kfolds nested list at {}kfolds.json.'.format(path))

    print('Completed {}-fold cross-validation.'.format(n_splits))
    return kfolds


if __name__ == '__main__':

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Script started on {}'.format(time.ctime()))

    # clean slate output_dr
    if os.path.exists(output_dr):
        shutil.rmtree(output_dr)
    if not os.path.exists(output_dr):
        os.mkdir(output_dr)

    main()

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Script finished on {}.'.format(time.ctime()))
