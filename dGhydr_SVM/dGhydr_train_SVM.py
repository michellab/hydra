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
from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args

# SVM:
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from scipy import stats

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
startpoint_BO = np.inf  # Point to consider top-performing model from (MAE/MAD); 1.0 = no improvement on test-set variance

# KFold parameters:
n_splits = 5  # Number of K-fold splits
random_state = 2  # Random number seed


def main():

    # initiate log file
    logging.basicConfig(filename= output_dr + 'train_logfile.txt',
                        filemode='a',
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)

    logging.info('Starting dGhydr_{}.py.'.format(model_type))

    # Load in dataset.
    train_df = pd.read_csv(datasets_dr + 'train_data.csv', index_col='Unnamed: 0')

    # training
    kfolds = split_dataset(train_df, n_splits, random_state)
    run_regressor(kfolds)

    logging.info('Finished dGhydr_{}.py.'.format(model_type))


def regressor(fold, fold_num):
    """
    Perofmrs:
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

    # nested list containing all models
    all_models = []

    # retrieve datasets
    train_X = fold[0][0].values
    validate_X = fold[0][1].values
    train_y = fold[1][0].values
    validate_y = fold[1][1].values

    # validate label pandas series for statistical analysis
    validate_y_df = fold[1][1]

    # Set hyperparameter ranges, append to list:
    # skopt.space.Catagorical
    dim_param_C = Categorical(categories=list(np.logspace(-3, 2, 6, dtype="float32")), name="param_C")
    dim_param_gamma = Categorical(categories=list(np.logspace(-3, 2, 6, dtype="float32")), name="param_gamma")
    dim_param_epsilon = Categorical(categories=list(np.logspace(-3, 2, 6, dtype="float32")), name="param_epsilon")

    dimensions = [dim_param_C, dim_param_gamma, dim_param_epsilon]

    @use_named_args(dimensions=dimensions)
    def fitness(param_C, param_gamma, param_epsilon):
        """Create svr with """

        # define SVR classifier
        regr = SVR(gamma=param_gamma, C=param_C, epsilon=param_epsilon)

        # fit and validate model
        regr.fit(train_X, train_y)
        predicted_y = regr.predict(validate_X)

        # calculate some statistics on validate set:
        MAE = mean_absolute_error(validate_y, predicted_y)
        MAD_validate = validate_y_df.mad()

        MAEMAD = MAE / MAD_validate
        print('Fold {} MAE/MAD: {}.'.format(fold_num, MAEMAD))

        valdt_ID_lst = validate_y_df.index.tolist()
        valdt_y_lst = validate_y_df.values.tolist()

        slope, intercept, r_value, p_value, std_err = stats.linregress(predicted_y, valdt_y_lst)
        tau, p_value = stats.kendalltau(predicted_y, valdt_y_lst)

        # For plotting test set correlations:
        tuples_result = list(zip(valdt_ID_lst, valdt_y_lst, predicted_y))
        # [ ..., [ID, [valdt_y], predicted_y], ... ]
        nested_lst_result = [list(elem) for elem in tuples_result]

        # Append data with best performing model.
        # Data contains the MAE/MAD score, protein target, iteration,
        # tau, r value, the keras DNN model, the internal validation plot
        # and the data for external validation:

        startpoint_MAEMAD = startpoint_BO

        if MAEMAD < startpoint_MAEMAD:
            # keep track of models
            all_models.append([MAEMAD, fold_num, tau, r_value, nested_lst_result])

            # write all model files:
            with open(output_dr + 'fold_' + str(fold_num) + '_' + model_type + '_model.pickle', 'wb') as file:
                pickle.dump(regr, file)

            logging.info('Model saved at ' + output_dr + 'fold_' + str(fold_num) + '_' + model_type + '_model.pickle')

        return MAEMAD

    # Bayesian Optimisation to search through hyperparameter space.
    # Prior parameters were found by manual search and preliminary optimisation loops.
    # For running just dataset 13x500 calls, optimal hyperparameters from 150 calls were used as prior.
    default_parameters = [1.0, 1.0, 1.0]
    print('——————————————————————————————————————————')
    print('Created model, optimising hyperparameters...')

    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI',  # Expected Improvement.
                                n_calls=n_calls,
                                x0=default_parameters)

    print('Concluded optimal hyperparameters:')
    print('Fold {}: {}'.format(str(fold_num), search_result.x))
    logging.info('Finished training fold {}: {}.'.format(str(fold_num), search_result.x))

    print('——————————————————————————————————————————')

    # return skopt object and highest scoring model for this fold:
    return search_result, all_models[-1]


def run_regressor(kfolds):

    # Initiate empty dataframe to fill with cumulative minima.
    cumulative_MAEs = pd.DataFrame()
    cumulative_MAEtauR_df = pd.DataFrame()
    MAEtauR_results_per_fold = [['Correlation Coefficient', 'Fold number', 'Correlation metric']]

    fold_num = 1
    models = []

    for fold in kfolds:
        # reset MAEMAD startpoint per replicate:
        OptimizeResult, top_model = regressor(fold, fold_num)

        models.append(top_model)

        # construct, cummin and concatenate results of this fold to the other folds in the loop:
        split_columns = {
            'Fold': int(fold_num),
            'MAE/MAD': OptimizeResult.func_vals}

        # construct individual fold result dataframe using the dictionary method
        fold_result_df = pd.DataFrame(split_columns).cummin()
        cumulative_MAEs = pd.concat([cumulative_MAEs, fold_result_df])

        # retrieve statistics for this replicate:
        tau = top_model[2]
        r_value = top_model[3]
        MAE = top_model[0]

        MAEtauR_results_per_fold.append([r_value, fold_num, 'Pearsons-r'])
        MAEtauR_results_per_fold.append([tau, fold_num, 'Kendalls-tau'])
        MAEtauR_results_per_fold.append([MAE, fold_num, 'MAE/MAD'])

        fold_num += 1

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Finished training')

    # models: [MAEMAD, fold_num, tau, r_value, nested_lst_result]
    # nested_lst_results: [ ..., [ID, [valdt_y], predicted_y], ... ]

    # make ensemble of best models; pick n replicates' top performing models:
    # explaination of key=lambda:
    # https://stackoverflow.com/questions/8966538/syntax-behind-sortedkey-lambda
    all_models = sorted(models, key=lambda x: x[0])

    for model in all_models:

        internal_fold_num = model[1]
        internal_validation = model[4]

        # For each model, write internal validation to file
        with open(output_dr + 'fold_' + str(internal_fold_num) + '_internal_validation.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Experimental dGoffset (kcal/mol)', 'Predicted dGoffset (kcal/mol)'])
            for row in internal_validation:
                writer.writerow(row)

    MAEtauR_df = pd.DataFrame(MAEtauR_results_per_fold[1:], columns=MAEtauR_results_per_fold[0])
    cumulative_MAEtauR_df = pd.concat([cumulative_MAEtauR_df, MAEtauR_df])

    # Save to CSV
    save_loc1 = output_dr + 'dGoffset_' + model_type + '_MAEtauR_outputs.csv'
    save_csv(cumulative_MAEtauR_df, save_loc1)

    save_loc2 = output_dr + 'dGoffset_' + model_type + '_BO_MAE.csv'
    save_csv(cumulative_MAEs, save_loc2)

    return cumulative_MAEs


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


def save_csv(dataframe, pathname):

    if os.path.exists(pathname):
        os.remove(pathname)
        dataframe.to_csv(path_or_buf=pathname, index=True)
        print('Existing file overwritten.')
    else:
        dataframe.to_csv(path_or_buf=pathname, index=True)
    print('Completed writing {}.csv.'.format(pathname))


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
