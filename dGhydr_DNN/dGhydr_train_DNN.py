# General:
import pandas as pd
import numpy as np
import os
import csv
import subprocess
import time
import shutil
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import statistics
import pickle
import logging
import json

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K

# SciKit-Optimise:
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

# SVM:
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# RDKit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolfiles, rdMolDescriptors
from rdkit.Chem import SDMolSupplier, Descriptors, Crippen, Lipinski, Fragments
from rdkit import DataStructs

# Misc.:
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from scipy import stats
import statistics
import pickle
from mordred import Calculator, descriptors

# global variables
path = './'
datasets_dr = '../datasets/'
SDF_dr = datasets_dr + 'sdffiles/'
freesolv_loc = datasets_dr + 'database.txt'
train_dr = path + 'train_dr/'
test_dr = path + 'test_dr/'
output_dr = path + 'output/'
figures_dr = path + 'figures/'

#
offset_col_name = 'dGoffset (kcal/mol)'
# SAMPl4_Guthrie experimental reference in FreeSolv.
SAMPL4_Guthrie_ref = 'SAMPL4_Guthrie'
# Experimental reference column name.
exp_ref_col = 'experimental reference (original or paper this value was taken from)'

# set data processing configurations:
model_type = 'DNN'

PCA_threshold = 0.95  # Keeps n dimensions for x variance explained
replicates = 30  # Number of replicates per subject model
n_calls = 40  # Number of Bayesian optimisation loops for hyperparameter optimisation, 40 is best for convergence, > 60 scales to very expensive
startpoint_BO = np.inf  # Point to consider top-performing model from (MAE/MAD); 1.0 = no improvement on test-set variance
ensemble_size = 10  # Amount of top-scoring models to retain per fold-dataset combination

# KFold parameters:
n_splits = 5  # Number of K-fold splits
random_state = 2  # Random number seed

# Load in FreeSolve
freesolv_df = pd.read_csv(freesolv_loc, sep='; ', engine='python')


def main():

    # initiate log file
    logging.basicConfig(filename= output_dr + 'training_logfile.txt',
                        filemode='a',
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)
    logging.info('Starting dGhydr_training_{}.py.'.format(model_type))

    # feature generation
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Generating features...')
    logging.info('Generating features...')
    mordred_df = get_descriptors()
    FP_df = get_fingerprints()
    compiled_X_df = compile_features(mordred_df, FP_df)
    numeric_X = check_dataframe_is_numeric(compiled_X_df)
    float_X = convert_to_float(numeric_X)
    normalised_X = normalise_and_split_datasets(float_X)
    reduced_X = reduce_features(normalised_X, PCA_threshold)
    print('Feature generation successful.')
    logging.info('Finished generating filters.')

    # label generation
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Generating labels...')
    logging.info('Generating labels...')
    true_y = get_labels()
    print('Label geneartion successful.')
    print('Finished generating lables.')
    logging.info('Finished generating labels.')

    # complete datasets ready for training
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Preparing for training...')
    full_dataset = get_full_dataset(reduced_X, true_y)
    train_df, test_df = separate_train_test(full_dataset)
    print('Pre training complete.')

    # training
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Training...')
    logging.info('Training started...')
    kfolds = split_dataset(train_df, n_splits, random_state)
    run_regressor(kfolds)
    print('Training complete.')
    logging.info('Training complete.')

    logging.info('Finished dGhydr_train_{}.py.'.format(model_type))

    ####################### set two main report colour themes and one extra #######################

    ####################### FG counter to be added to end of plot_fingerprint_similarity #######################


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
    cumulative_MAEtauR_df_save_loc = output_dr + 'dGoffset_' + model_type + '_MAEtauR_outputs.csv'

    if os.path.exists(cumulative_MAEtauR_df_save_loc):
        os.remove(cumulative_MAEtauR_df_save_loc)
        cumulative_MAEtauR_df.to_csv(path_or_buf=cumulative_MAEtauR_df_save_loc, index=True)
        print('Existing file overwritten.')
    else:
        cumulative_MAEtauR_df.to_csv(path_or_buf=cumulative_MAEtauR_df_save_loc, index=True)

    print('Completed writing cumulative MAE, tau and R to CSV.')

    # Save to CSV
    cumulative_MAE_save_loc = output_dr + 'dGoffset_' + model_type +  '_BO_MAE.csv'

    if os.path.exists(cumulative_MAE_save_loc):
        os.remove(cumulative_MAE_save_loc)
        cumulative_MAEs.to_csv(path_or_buf=cumulative_MAE_save_loc, index=True)
        print('Existing file overwritten.')
    else:
        cumulative_MAEs.to_csv(path_or_buf=cumulative_MAE_save_loc, index=True)

    print('Completed writing cumulative MAE, tau and R to CSV.')
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


def save_csv(dataframe, filename):

    save_loc = path + filename +'.csv'
    if os.path.exists(save_loc):
        os.remove(save_loc)
        dataframe.to_csv(path_or_buf=save_loc, index=True)
        print('Existing file overwritten.')
    else:
        dataframe.to_csv(path_or_buf=save_loc, index=True)
    print('Completed writing {}.csv.'.format(filename))


def save_sdf(ID_lst, dr_name):

    # Create directory
    new_dr = path + dr_name + '/'
    if os.path.isdir(new_dr):
        shutil.rmtree(new_dr)
        print('Existing directory overwritten.')
        os.mkdir(new_dr)
    else:
        os.mkdir(new_dr)

    for entry in ID_lst:
        sdf = entry + '.sdf'
        shutil.copyfile(SDF_dr + sdf, new_dr + sdf)

    # Check the number of ligands found is correct.
    print('Number of entires in {}: {}'.format(dr_name, len(glob.glob(new_dr + '*.sdf'))))


def separate_train_test(full_dataset):

    # List comprehension for all non-SAMPL4_Guthrie entires.
    train_IDs = [freesolv_df.iloc[i][0]
                 for i in range(len(freesolv_df))
                 if freesolv_df.loc[i, exp_ref_col] != SAMPL4_Guthrie_ref]

    # List comprehension for all SAMPL4_Guthrie entires.
    test_IDs = [freesolv_df.iloc[i][0]
                for i in range(len(freesolv_df))
                if freesolv_df.loc[i, exp_ref_col] == SAMPL4_Guthrie_ref]

    print('Creating training set...')
    train_df = full_dataset.drop(test_IDs)
    save_csv(train_df, 'train_data')
    save_sdf(train_IDs, 'train_dr')

    print('Creating testing set...')
    test_df = full_dataset.drop(train_IDs)
    save_csv(test_df, 'test_data')
    save_sdf(test_IDs, 'test_dr')

    return train_df, test_df


def get_full_dataset(feature_df, label_df):

    print('Generating full dataset...')
    full_dataset = pd.concat([feature_df, label_df], axis=1, sort=False)

    # Save to CSV
    save_loc = path + 'full_dataset.csv'

    if os.path.exists(save_loc):
        os.remove(save_loc)
        full_dataset.to_csv(path_or_buf=save_loc, index=False)
        print('Existing file overwritten.')
    else:
        full_dataset.to_csv(path_or_buf=save_loc, index=False)

    print('Completed genearting full dataset and writing to CSV.')
    return full_dataset


def get_labels():

    # Load in FreeSolve
    freesolv_df = pd.read_csv(freesolv_loc, sep='; ', engine='python')

    # Column names
    freesolv_ID = freesolv_df.loc[:, 'compound id (and file prefix)']
    exp_val = freesolv_df.loc[:, 'experimental value (kcal/mol)']
    exp_err = freesolv_df.loc[:, 'experimental uncertainty (kcal/mol)']
    calc_val = freesolv_df.loc[:, 'Mobley group calculated value (GAFF) (kcal/mol)']
    calc_err = freesolv_df.loc[:, 'calculated uncertainty (kcal/mol)']

    # New nested list containing IDs and offsets
    offsets = []
    for name, exp, err1, calc, err2 in zip(freesolv_ID, exp_val, exp_err, calc_val, calc_err):
        offset = exp - calc
        error = (err1 ** 2 + err2 ** 2) ** 0.5
        offsets.append([name, offset, round(error, 3)])

    # Experimental offsets with uncertainties
    exp_offset_with_errors_df = pd.DataFrame(offsets, columns=['ID', 'dGoffset (kcal/mol)', 'uncertainty (kcal/mol)'])

    # Experimental offsets only
    exp_offset = exp_offset_with_errors_df.drop(columns=['uncertainty (kcal/mol)'])
    exp_offset = exp_offset.set_index('ID')

    # Save to CSV
    save_loc = path + 'labels_y/exp_labels.csv'

    if os.path.exists(save_loc):
        os.remove(save_loc)
        exp_offset.to_csv(path_or_buf=save_loc, index=False)
        print('Existing file overwritten.')
    else:
        exp_offset.to_csv(path_or_buf=save_loc, index=False)

    print('Completed genearting labels and writing to CSV.')
    return exp_offset


def reduce_features(normalised_collection, pca_threshold):
    print('Computing PCA, reducing features up to ' + str(round(pca_threshold * 100, 5)) + '% VE...')
    training_data = normalised_collection

    # Initialise PCA object, keep components up to x% variance explained:
    PCA.__init__
    pca = PCA(n_components=pca_threshold)

    # Fit to and transform training set:
    train_post_pca = pd.DataFrame(pca.fit_transform(training_data))

    # Reset column names to PCX
    PCA_col = np.arange(1, len(train_post_pca.columns) + 1).tolist()
    PCA_col = ['PC' + str(item) for item in PCA_col]
    train_post_pca.columns = PCA_col
    train_post_pca.index = training_data.index

    print('Number of PCA features after reduction: ' + str(len(train_post_pca.columns)))

    # pickle pca object to file so that external test sets can be transformed accordingly
    # (see https://stackoverflow.com/questions/42494084/saving-large-data-set-pca-on-disk
    # -for-later-use-with-limited-disc-space)
    # pickle.dump(pca, open('./opt_output/pca_trainingset.p', 'wb'))

    def recovery_pc(normalised_collection, pca_threshold):
        print('Computing PCA, reducing features up to ' + str(round(pca_threshold * 100, 5)) + '% VE...')
        training_data = normalised_collection

        # normalise data
        data_scaled = pd.DataFrame(preprocessing.scale(training_data), columns=training_data.columns)

        # Initialise PCA object, keep components up to x% variance explained:
        PCA.__init__
        pca = PCA(n_components=pca_threshold)
        pca.fit_transform(data_scaled)

        index = list(range(1, 111 + 1))
        index = ['PC{}'.format(x) for x in index]

        return_df = pd.DataFrame(pca.components_, columns=data_scaled.columns, index=index)

        return return_df

    # adapted from https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
    recovered_pc = recovery_pc(normalised_collection, PCA_threshold)

    # list of column names with highest value in each row
    recovered_pc_max = recovered_pc.idxmax(axis=1)

    # recovery 'PCX' indexing
    pc_index = recovered_pc_max.index.tolist()

    # write feature names to list
    pc_feature = recovered_pc_max.values.tolist()

    # write to df
    recovered_pc_dict = {'PCX': pc_index, 'Highest contributing feature': pc_feature}
    recovered_pc_df = pd.DataFrame(recovered_pc_dict)

    # Save to CSV
    save_loc = path + 'recovered_PCs.csv'

    if os.path.exists(save_loc):
        os.remove(save_loc)
        recovered_pc_df.to_csv(path_or_buf=save_loc, index=False)
    else:
        recovered_pc_df.to_csv(path_or_buf=save_loc, index=False)

    print('Completed writing recovered PCs to CSV.')
    return train_post_pca  # return list with test_post_pca when needed


def normalise_and_split_datasets(dataframe):

    # Calculate statistics, compute Z-scores, clean:
    print('Normalising dataframe...')
    stats = dataframe.describe()
    stats = stats.transpose()

    def norm(x):
        return (x - stats['mean']) / stats['std']

    # Normalise and return separately:
    normed_data = norm(dataframe).fillna(0).replace([np.inf, -np.inf], 0.0)

    print('Completed normalising dataframe.')
    return normed_data


def convert_to_float(dataframe):

    print('Converting dataframe to float...')
    float_df = dataframe.apply(pd.to_numeric).astype(float).sample(frac=1)
    float_df = float_df.rename(columns={'dGhydr (kcal/mol)': 'dGoffset (kcal/mol)'})

    print('Completed converting dataframe to flaot.')
    return float_df


def check_dataframe_is_numeric(dataframe):
    """Iterate over all columns and check if numeric.

    Returns:
    New DataFrame with removed"""

    columns_dropped = 0
    columns_dropped_lst = []

    print('Checking dataframe is numeric...')
    for col in dataframe.columns:
        for index, x in zip(dataframe.index, dataframe.loc[:, col]):
            try:
                float(x)
            except ValueError:
                columns_dropped_lst.append([col, index, x])
                columns_dropped += 1
                dataframe = dataframe.drop(columns=col)
                break

    # save to CSV
    dropped_col_df = pd.DataFrame(columns_dropped_lst, columns=['column dropped', 'at ID', 'non-numeric value'])
    save_loc = path + 'features_X/dropped_features.csv'

    if os.path.exists(save_loc):
        os.remove(save_loc)
        dropped_col_df.to_csv(path_or_buf=save_loc, index=False)
        print('Existing file overwritten.')
    else:
        dropped_col_df.to_csv(path_or_buf=dropped_col_save_loc, index=False)

    print('Number of columns dropped:', (columns_dropped))
    print('Completed writing dropped columns to CSV.')
    return dataframe


def compile_features(df1, df2):

    compiled_df = df1.set_index('ID')
    compiled_df = compiled_df.join(df2.set_index('ID'), on='ID')
    # compiled_df = compiled_df.set_index('ID')

    print('Completed joining dataframes.')
    return compiled_df


def get_fingerprints():

    save_loc = path + 'features_X/fingerprints_output/fingerprints.csv'

    if os.path.exists(save_loc):
        FP_df = pd.read_csv(save_loc)
        print('Calculated fingerprints loaded in.')
    else:
        FP_table = []
        for sdf in glob.glob(SDF_dr + '*.sdf'):

            FP_row = []

            # Append ligand ID
            FP_row.append(sdf.strip(SDF_dr).strip('*.sdf'))

            # Setup fingerprint
            mol = Chem.rdmolfiles.SDMolSupplier(sdf)[0]
            mol.UpdatePropertyCache(strict=False)

            # Calculate fingerprint
            print('Calculating fingerprints...')
            FP = rdMolDescriptors.GetHashedAtomPairFingerprint(mol, 256)
            for x in list(FP):
                FP_row.append(x)

            FP_table.append(FP_row)

        # Column names
        ID_col = ['ID']
        FP_col = np.arange(0, 256).tolist()
        FP_col = [ID_col.append("pfp" + str(item)) for item in FP_col]

        FP_df = pd.DataFrame(FP_table, columns=ID_col)

        print('Completed calculating fingerprints.')

    return FP_df


def get_descriptors():

    save_loc = path + 'features_X/mordred_descriptor_output/mordred_descriptors.csv'

    if os.path.exists(save_loc):
        mordred_df = pd.read_csv(save_loc)
        print('Calculated Mordred descriptors loaded in.')
    else:
        descriptors_raw = open(datasets_dr + 'all_mordred_descriptors.txt', 'r')
        descriptors_raw_list = [line.split('\n') for line in descriptors_raw.readlines()]
        descriptors_list = [desc[0] for desc in descriptors_raw_list]
        print('Number of descriptors:', str(len(descriptors_list)))

        # set up feature calculator
        print('Calculating Mordred descriptors...')
        calc = Calculator(descriptors, ignore_3D=False)

        # Supply SDF
        suppl = [sdf for sdf in glob.glob(SDF_dr + '*.sdf')]

        # Empty DataFrame containing only descriptor names as headings
        mordred_df = pd.DataFrame(columns=descriptors_list)

        # generate features
        ID_lst = []
        for mol in suppl:
            ID = mol.strip(SDF_dr)
            ID_lst.append(ID)
            feat = calc.pandas(Chem.SDMolSupplier(mol))
            mordred_df = mordred_df.append(feat, ignore_index=True, sort=False)

        # Insert IDs as new column with column index = 0
        mordred_df.insert(0, 'ID', ID_lst)

        print('Finished calculating Mordred descriptors.')

        if os.path.exists(save_loc):
            os.remove(save_loc)
            mordred_df.to_csv(path_or_buf=save_loc, index=False)
            print('Existing file overwritten.')
        else:
            mordred_df.to_csv(path_or_buf=save_loc, index=False)

        print('Completed writing all calculated mordred descriptors to CSV.')

    return mordred_df


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
