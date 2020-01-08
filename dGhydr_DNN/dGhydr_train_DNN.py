# General:
import pandas as pd
import numpy as np
import os
import csv
import time
import shutil
import pickle
import logging

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from scipy import stats

# Sklearn
from skopt import gp_minimize
from skopt.space import Categorical, Integer
from skopt.utils import use_named_args
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Path variables:
path = './'
datasets_dr = '../datasets/'
SDF_dr = datasets_dr + 'sdffiles/'
output_dr = path + 'output/'
if not os.path.exists(output_dr):
    os.mkdir(output_dr)
figures_dr = path + 'figures/'
if not os.path.exists(output_dr):
    os.mkdir(figures_dr)

# Global variables:
model_type = 'DNN'
offset_col_name = 'dGoffset (kcal/mol)'

# set data processing configurations:
n_calls = 40  # Number of Bayesian optimisation loops for hyperparameter optimisation, 40 is best for convergence, > 60 scales to very expensive
startpoint_BO = np.inf  # Point to consider top-performing model from (MAE/MAD); 1.0 = no improvement on test-set variance

# KFold parameters:
n_splits = 5  # Number of K-fold splits
random_state = 2  # Random number seed


def main():

    # initiate log file
    logging.basicConfig(filename= output_dr + 'training_logfile.txt',
                        filemode='a',
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)
    logging.info('Starting dGhydr_training_{}.py.'.format(model_type))

    # Load in dataset.
    train_df = pd.read_csv(datasets_dr + 'train_data.csv', index_col='Unnamed: 0')

    # training
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Training...')
    logging.info('Training started...')
    kfolds = split_dataset(train_df, n_splits, random_state)
    run_regressor(kfolds)
    print('Training complete.')
    logging.info('Training complete.')

    logging.info('Finished dGhydr_train_{}.py.'.format(model_type))


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

    tf.keras.backend.clear_session()
    # tf.reset_default_graph()

    # Display training progress by printing a single dot per epoch:
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            print('.', end='')

    # nested list containing all models
    all_models = []

    # Set early stopping variable:
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=20,
        verbose=0)

    # retrieve datasets
    train_X = fold[0][0].values
    validate_X = fold[0][1].values
    train_y = fold[1][0].values
    validate_y = fold[1][1].values

    # validate label pandas series for statistical analysis
    validate_y_df = fold[1][1]

    # Build keras DNN using global params:
    def create_model(
            num_dense_layers_base,
            num_dense_nodes_base,
            num_dense_layers_end,
            num_dense_nodes_end,
            activation,
            adam_b1,
            adam_b2,
            adam_eps,
            num_batch_size):

        model = keras.Sequential()

        # Add input layer of length of the dataset columns:
        # model.add(keras.layers.Dense(len(train_X.columns), input_shape=[len(train_X.keys())]))
        model.add(keras.layers.Dense(len(fold[0][0].columns), input_shape=[len(fold[0][0].keys())]))

        # Generate n number of hidden layers (base, i.e. first layers):
        for i in range(num_dense_layers_base):
            model.add(keras.layers.Dense(num_dense_nodes_base,
                                         activation=activation
                                         ))

        # Generate n number of hidden layers (end, i.e. last layers):
        for i in range(num_dense_layers_end):
            model.add(keras.layers.Dense(num_dense_nodes_end,
                                         activation=activation
                                         ))

        # Add output layer:
        model.add(keras.layers.Dense(1, activation=keras.activations.linear))

        optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=adam_b1, beta_2=adam_b2, epsilon=adam_eps)

        model.compile(
            loss='mae',
            optimizer=optimizer,
            metrics=["mae"]
        )
        return model

    # Set hyperparameter ranges, append to list:
    dim_num_dense_layers_base = Integer(low=1, high=2, name='num_dense_layers_base')
    dim_num_dense_nodes_base = Categorical(categories=list(np.linspace(5, 261, 10, dtype=int)),
                                           name='num_dense_nodes_base')
    dim_num_dense_layers_end = Integer(low=1, high=2, name='num_dense_layers_end')
    dim_num_dense_nodes_end = Categorical(categories=list(np.linspace(5, 261, 10, dtype=int)),
                                          name='num_dense_nodes_end')

    # dim_activation = Categorical(categories=[tf.keras.activations.relu], name='activation')
    dim_adam_b1 = Categorical(categories=list(np.linspace(0.8, 0.99, 11)), name='adam_b1')
    dim_adam_b2 = Categorical(categories=list(np.linspace(0.8, 0.99, 11)), name='adam_b2')
    dim_adam_eps = Categorical(categories=list(np.linspace(0.0001, 0.5, 11)), name='adam_eps')
    dim_num_batch_size = Categorical(categories=list(np.linspace(32, 128, 7, dtype=int)), name='num_batch_size')

    dimensions = [
        dim_num_dense_layers_base,
        dim_num_dense_nodes_base,
        dim_num_dense_layers_end,
        dim_num_dense_nodes_end,
        dim_adam_b1,
        dim_adam_b2,
        dim_adam_eps,
        dim_num_batch_size]

    @use_named_args(dimensions=dimensions)
    def fitness(
            num_dense_layers_base,
            num_dense_nodes_base,

            num_dense_layers_end,
            num_dense_nodes_end,
            adam_b1,
            adam_b2,
            adam_eps,
            num_batch_size):

        # Create the neural network with these hyper-parameters:
        regr = create_model(
            num_dense_layers_base=num_dense_layers_base,
            num_dense_nodes_base=num_dense_nodes_base,
            num_dense_layers_end=num_dense_layers_end,
            num_dense_nodes_end=num_dense_nodes_end,
            activation=tf.keras.activations.relu,
            adam_b1=adam_b1,
            adam_b2=adam_b2,
            adam_eps=adam_eps,
            num_batch_size=num_batch_size)

        # print('Fitting model..')
        history = regr.fit(
            train_X, train_y,
            epochs=1000,
            validation_data=(validate_X, validate_y),
            verbose=0,
            callbacks=[
                early_stopping,
                # PrintDot(),			# uncomment for verbosity on epochs
            ],
            batch_size=121)

        # calculate some statistics on test set:
        prediction = regr.predict(validate_X)
        predicted_y = [item[0] for item in prediction]

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        # MAE = hist['val_mean_absolute_error'].tail(10).mean()
        MAE = mean_absolute_error(validate_y, predicted_y)
        # MAD_testset = validate_y.mad()
        MAD_validate = validate_y_df.mad()

        MAEMAD = MAE / MAD_validate
        print('Fold {} MAE/MAD: {}.'.format(fold_num, MAEMAD))

        valdt_ID_lst = validate_y_df.index.tolist()
        valdt_y_lst = validate_y_df.values.tolist()

        slope, intercept, r_value, p_value, std_err = stats.linregress(predicted_y, valdt_y_lst)
        tau, p_value = stats.kendalltau(predicted_y, valdt_y_lst)

        # for plotting test set correlations:
        tuples_result = list(zip(valdt_ID_lst, valdt_y_lst, predicted_y))
        # [ ..., [ID, [valdt_y], predicted_y], ... ]
        nested_lst_result = [list(elem) for elem in tuples_result]

        startpoint_MAEMAD = startpoint_BO
        if MAEMAD < startpoint_MAEMAD:
            startpoint_MAEMAD = MAEMAD
            # keep track of models
            all_models.append([MAEMAD, fold_num, tau, r_value, regr, hist, nested_lst_result])

            # # write all model files:
            # # Slightly hacky but TF's backend voids model parameters when the model is saved as a variable
            # # in order to retain the top performing model. From these temporary model files, all but the
            # # top-performing model will be deleted from the system at the end of this script.

            # regr.save_weights(output_dr + 'fold_' + str(fold_num) + '_' + model_type + '_model.h5')
            # with open(output_dr + 'fold_' + str(fold_num) + '_' + model_type + '_model.' + model_type.lower(), 'w') as file:
            #     pickle.dump(regr, file)

            # https://www.tensorflow.org/tutorials/keras/save_and_load
            regr.save(output_dr + 'fold_' + str(fold_num) + '_' + model_type + '_model.h5')

        return MAEMAD

    # Bayesian Optimisation to search through hyperparameter space.
    # Prior parameters were found by manual search and preliminary optimisation loops.
    # For running just dataset 13x500 calls, optimal hyperparameters from 150 calls were used as prior.
    default_parameters = [2, 33, 1, 90, 0.971, 0.895, 1.0000e-04, 112]
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

    # initiate empty dataframe to fill with cumulative minima
    cumulative_MAEs = pd.DataFrame()
    cumulative_MAEtauR_df = pd.DataFrame()
    mae_results_per_fold = [['Subject', 'MAE', 'Replicate']]
    MAEtauR_results_per_fold = [['Correlation Coefficient', 'Fold number', 'Correlation metric']]

    fold_num = 1
    models = []

    for fold in kfolds:
        # run svr:
        # reset MAEMAD startpoint per replicate:
        OptimizeResult, top_model = regressor(fold, fold_num)
        print('Fold {} copleted training.'.format(fold_num))

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

        # write update to log file:
        # with open(output_dr + 'logfile.txt', 'a') as file:
        #     writer = csv.writer(file, delimiter='\t')
        #     writer.writerow(['Finished fold', fold_num, 'at', str(time.ctime())])

        fold_num += 1

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Finished training')

    # models: [MAEMAD, fold_num, tau, r_value, model, hist, nested_lst_result]
    # nested_lst_results: [ ..., [ID, [valdt_y], predicted_y], ... ]

    # make ensemble of best models; pick n replicates' top performing models:
    # explaination of key=lambda:
    # https://stackoverflow.com/questions/8966538/syntax-behind-sortedkey-lambda
    all_models = sorted(models, key=lambda x: x[0])

    for model in all_models:

        internal_fold_num = model[1]
        internal_validation = model[6]

        # For each model, write internal validation to file
        with open(output_dr + 'fold_' + str(internal_fold_num) + '_internal_validation.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Experimental dGoffset (kcal/mol)', 'Predicted dGoffset (kcal/mol)'])
            for row in internal_validation:
                writer.writerow(row)

    MAEtauR_df = pd.DataFrame(MAEtauR_results_per_fold[1:], columns=MAEtauR_results_per_fold[0])
    cumulative_MAEtauR_df = pd.concat([cumulative_MAEtauR_df, MAEtauR_df])

    # Save to CSV
    save_loc_1 = output_dr + 'dGoffset_' + model_type + '_MAEtauR_outputs.csv'
    save_csv(cumulative_MAEtauR_df, save_loc_1)

    # Save to CSV
    save_loc_2 = output_dr + 'dGoffset_' + model_type +  '_BO_MAE.csv'
    save_csv(cumulative_MAEs, save_loc_2)

    return cumulative_MAEs


def split_dataset(dataset, n_splits, random_state):
    """KFold implementation for pandas DataFrame.
    (https://stackoverflow.com/questions/45115964/separate-pandas-dataframe-using-sklearns-kfold)"""

    print('Performing {}-fold cross-validation...'.format(n_splits))

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

    print('Pickled kfolds nested list.')
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
    print('Script finished.')
