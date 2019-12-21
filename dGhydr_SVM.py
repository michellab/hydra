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
path = './absolute_dGoffset/'
SDF_dr = './datasets/sdffiles/'
freesolv_loc = './datasets/database.txt'
train_dr = path + 'train_dr/'
test_dr = path + 'test_dr/'
output_dr = path + 'output'

#
offset_col_name = 'dGoffset (kcal/mol)'
# SAMPl4_Guthrie experimental reference in FreeSolv.
SAMPL4_Guthrie_ref = 'SAMPL4_Guthrie'
# Experimental reference column name.
exp_ref_col = 'experimental reference (original or paper this value was taken from)'

# set data processing configurations:
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

    # feature generation
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Generating features...')
    mordred_df = get_descriptors()
    FP_df = get_fingerprints()
    compiled_X_df = compile_features(mordred_df, FP_df)
    numeric_X = check_dataframe_is_numeric(compiled_X_df)
    float_X = convert_to_float(numeric_X)
    normalised_X = normalise_and_split_datasets(float_X)
    reduced_X = reduce_features(normalised_X, PCA_threshold)
    print('Feature generation successful.')

    # label generation
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Generating labels...')
    true_y = get_labels()
    print('Label geneartion successful.')

    # complete datasets ready for training
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Preparing for training...')
    full_dataset = get_full_dataset(reduced_X, true_y)
    train_df, test_df = separate_train_test(full_dataset)
    print('Pre training complete.')

    # training
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Training...')
    kfolds = split_dataset(train_df, n_splits, random_state)
    cumulative_MAE_df = run_svr(kfolds)
    plot_convergence(cumulative_MAE_df, 40)
    print('Training complete.')

    # testing
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Testing...')
    predicted_offset, predicted_offset_mae = predict_offset(test_df)
    corrected_hydr, corrected_hydr_mae, mobley_hydr_mae = correct_hydration(predicted_offset)
    print('Finished external testing.')

    # plot graphs
    plot_scatter(predicted_offset,
                 [1, 'Experimental dGoffset (kcal/mol)'],
                 [2, 'Averaged predicted dGoffset (kcal/mol)'],
                 title='External test set dGoffsets',
                 MAE=predicted_offset_mae)

    plot_scatter(corrected_hydr,
                 [1, 'Experimental dGhydr (kcal/mol)'],
                 [2, 'Calculated dGhydr (kcal/mol)'],
                 title='Original Mobley calculated dGhydr',
                 MAE=mobley_hydr_mae)

    plot_scatter(corrected_hydr,
                 [1, 'Experimental dGhydr (kcal/mol)'],
                 [4, 'Corrected calculated dGhydr (kcal/mol)'],
                 title='External test set corrected calculated dGhydr',
                 MAE=corrected_hydr_mae)

    # fingerprint similarity
    print('Plotting fingerprint similarity plots...')
    plot_fingerprint_similarity(corrected_hydr, train_df, test_df, kfolds, 'highest')
    plot_fingerprint_similarity(corrected_hydr, train_df, test_df, kfolds, 'lowest')

    ####################### statistical metric for number of ligands > 95% #######################


def plot_fingerprint_similarity(corrected_hydr, train_df, test_df, kfolds, mode):

    # retrieve best and worst predicted dGhydr
    corr_AE_sort_df = corrected_hydr.sort_values(by='Corrected calculated dGhydr absolute error (kcal/mol)')
    target_df = pd.concat([corr_AE_sort_df.head(), corr_AE_sort_df.tail()])
    target_df = target_df.set_index('ID')

    test_ID = test_df.index.tolist()

    # selected external test set data
    if mode == 'highest':
        target_df = target_df.head(5)
    elif mode == 'lowest':
        target_df = target_df.tail(5)

    target_ID = target_df.index.tolist()
    target_MAE = target_df.iloc[:, 4].tolist()

    # initiate grid plot
    plt.figure()
    fig, axs = plt.subplots(nrows=n_splits,
                            ncols=len(target_ID),
                            figsize=(15, 6),
                            facecolor='w',
                            edgecolor='k',
                            sharex=True,
                            sharey=True)

    # add a big axis and hide frame
    fig.add_subplot(111, frameon=False)

    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Fingerprint similarity')
    plt.ylabel('Density')

    # iterate through all selected external test set data
    for j, mol, MAE in zip(range(len(target_ID)), target_ID, target_MAE):

        # target external test set entry fingerprint generation
        target_suppl = Chem.SDMolSupplier(test_dr + str(mol) + '.sdf')
        target_fp = Chem.RDKFingerprint(target_suppl[0])

        # iterate through each fold for selected targeted external test set entry
        for i, fold in zip(range(n_splits), kfolds):

            fold_num = j + 1

            # retrieve IDs
            train_IDs = fold[0][0].index.tolist()
            validate_IDs = fold[0][1].index.tolist()

            # retrieve SDFs
            train_suppl = [Chem.SDMolSupplier(train_dr + str(sdf) + '.sdf') for sdf in train_IDs]
            valdtn_suppl = [Chem.SDMolSupplier(train_dr + str(sdf) + '.sdf') for sdf in validate_IDs]

            # generate fingerprints
            train_fp = [Chem.RDKFingerprint(mol[0]) for mol in train_suppl]
            valdtn_fp = [Chem.RDKFingerprint(mol[0]) for mol in valdtn_suppl]

            # compute similarities
            train_similarity = [DataStructs.FingerprintSimilarity(target_fp, train_mol) for train_mol in train_fp]
            valdnt_similarity = [DataStructs.FingerprintSimilarity(target_fp, valdnt_mol) for valdnt_mol in valdtn_fp]

            # plot densities
            sns.distplot(train_similarity,
                         hist=False,
                         kde=True,
                         kde_kws={'linewidth': 2},
                         label='Train similarity',
                         ax=axs[i, j])

            sns.distplot(valdnt_similarity,
                         hist=False,
                         kde=True,
                         kde_kws={'linewidth': 2},
                         label='Validation similarity',
                         ax=axs[i, j])

            # remove all subplot legends and add global legend
            axs[i, j].get_legend().remove()
            handles, labels = axs[i, j].get_legend_handles_labels()
            # fig.legend(handles, labels, loc='lower center')
            fig.legend(handles, labels,
                       loc='lower center',
                       bbox_to_anchor=(0.5, 0.0),
                       ncol=2)

    # add row and column labels
    for x, row in enumerate(axs):
        for y, cell in enumerate(row):
            if x == 0:
                cell.xaxis.set_label_position('top')
                cell.set_xlabel('{}\nMAE: {} kcal/mol'.format(target_ID[y], round(target_MAE[y], 2)),
                                labelpad=10)
            if y == len(target_ID) - 1:
                cell.yaxis.set_label_position('right')
                cell.set_ylabel('Fold {}'.format(x + 1),
                                labelpad=25,
                                rotation=0)

    plt.tight_layout()

    filename = path + mode + '_fingerprint_similarity.png'
    plt.savefig(filename)
    print('Finished plotting figure {}.'.format(filename))


def correct_hydration(predicted_offset):

    # SAMPL4 Gurthrie df
    test_fs_df = freesolv_df.loc[freesolv_df.iloc[:, 7] == 'SAMPL4_Guthrie']

    # experimental dGhydr
    test_exp = test_fs_df.iloc[:, 3].tolist()

    # calculated dGhydr
    test_calc = test_fs_df.iloc[:, 5].tolist()

    # calculated dGhydr uncertainty
    test_calc_err = test_fs_df.iloc[:, 6].tolist()

    # corrected calculated Ghydr using predicted dGoffsets
    avg_offsets = predicted_offset['Averaged predicted dGoffset (kcal/mol)']
    corr_calc = [calc + err for calc, err in zip(test_calc, avg_offsets)]

    # calculated dGhydr absolute error
    calc_AE = [abs(exp - calc) for exp, calc in zip(test_exp, test_calc)]

    # corrected calculated dGhydr propogated absolute error
    # corr_AE = (err1**2 + err2**2)**0.5
    corr_AE = [abs(exp - calc) for exp, calc in zip(test_exp, corr_calc)]

    # create df
    corr_dict = {'ID': predicted_offset['ID'].tolist(),
                 'Experimental dGhydr (kcal/mol)': test_exp,
                 'Calculated dGhydr (kcal/mol)': test_calc,
                 'Calculated dGhydr absolute error (kcal/mol)': calc_AE,
                 'Corrected calculated dGhydr (kcal/mol)': corr_calc,
                 'Corrected calculated dGhydr absolute error (kcal/mol)': corr_AE}

    # corr_df = pd.DataFrame(corr_dict).round(2)
    corr_df = pd.DataFrame(corr_dict)

    # calculate MAEs
    calc_MAE = statistics.mean(calc_AE)
    print('Mobley calculated MAE: {} kcal/mol'.format(round(calc_MAE, 2)))
    corr_MAE = statistics.mean(corr_AE)
    print('Corrected calculated MAE: {} kcal/mol'.format(round(corr_MAE, 2)))

    return corr_df, corr_MAE, calc_MAE


def plot_scatter(dataframe, x_info, y_info, title, MAE):
    """x and y info are lists with fomrat [datatframe_index, axis label]."""

    # x and y data
    x = dataframe.iloc[:, x_info[0]]
    y = dataframe.iloc[:, y_info[0]]

    # plot scatter
    plt.figure()
    plt.scatter(x, y,
                color='black',
                s=8)

    # plot line of best fit
    # https://stackoverflow.com/questions/22239691/code-for-best-fit-straight-line-of-a-scatter-plot-in-python
    plt.plot(np.unique(x),
             np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),
             color='black',
             linewidth=1)

    # axis labels
    plt.xlabel(x_info[1])
    plt.ylabel(y_info[1])

    plt.title(title)

    # R-squared
    r2 = r2_score(x, y)

    # annotate with r-squared and MAE
    string = 'R-squared = {}\nMAE = {}'.format(round(r2, 4), round(MAE, 4))
    plt.annotate(string,
                 xy=(0, 1),
                 xytext=(12, -12),
                 va='top',
                 xycoords='axes fraction',
                 textcoords='offset points')

    filename = path + str(title).lower().replace(' ', '_') + '.png'
    plt.savefig(filename)
    print('Saved', filename)


def calc_mae(dataframe, model):

    model_df = dataframe.loc[dataframe['Model number'] == model]
    abs_err = model_df['Absolute error (kcal/mol)'].tolist()
    MAE = statistics.mean(abs_err)

    return MAE


def svr_predict(model_num, test_entry):

    with open(output_dr + '/fold_' + str(model_num) + '_SVM_model.svm', 'rb') as file:
        model = pickle.load(file)

    return model.predict(test_entry)


def predict_offset(test_set):

    print('Predicting offsets...')

    # load in testing set
    test_ID = test_set.index
    test_X = test_set.drop(columns='dGoffset (kcal/mol)').values
    test_y = test_set['dGoffset (kcal/mol)'].values

    # empty df for external testing results
    test_rst = pd.DataFrame()

    # peform prediction using each model
    num_models = list(range(1, n_splits + 1))
    for model in num_models:

        # call SVR prediction function
        svr_rst = svr_predict(model, test_X)

        # write results per fold into dictionary and load into df
        model_rst = {}
        model_rst['ID'] = test_ID
        model_rst['Model number'] = [model for i in range(41)]
        model_rst['Experimental dGoffset (kcal/mol)'] = test_y
        model_rst['Predicted dGoffset (kcal/mol)'] = svr_rst
        model_rst['Absolute error (kcal/mol)'] = abs(test_y - svr_rst)

        test_rst = pd.concat([test_rst, pd.DataFrame(model_rst)])

    # calculate MAE values
    MAE_lst = [calc_mae(test_rst, model) for model in num_models]
    print('MAE values between experimental and predicted dGoffset values:\n')
    for model, model_MAE in enumerate(MAE_lst): print('Model {} MAE: {} kcal/mol'.format(model + 1, round(model_MAE, 2)))
    print('\nAverage MAE: {} kcal/mol'.format(round(statistics.mean(MAE_lst), 2)))

    # round whole results dataframe to two decimal places
    test_rst = test_rst.round(2)

    # average predicted offset values
    prdt_offsets = [test_rst.loc[test_rst['Model number'] == model, 'Predicted dGoffset (kcal/mol)'].tolist()
                    for model in num_models]
    prdt_offsets = list(zip(*prdt_offsets))
    avg_offsets = [statistics.mean(offset_set) for offset_set in prdt_offsets]

    # write results to df
    avg_rst = {}

    avg_rst['ID'] = test_ID
    avg_rst['Experimental dGoffset (kcal/mol)'] = test_y
    avg_rst['Averaged predicted dGoffset (kcal/mol)'] = avg_offsets
    avg_rst['Absolute error (kcal/mol)'] = abs(test_y - avg_offsets)

    avg_rst_df = pd.DataFrame(avg_rst)

    # MAE
    print('MAE between experimental and averaged predicted dGoffsets:')
    test_offset_MAE = round(statistics.mean(abs(test_y - avg_offsets)), 2)
    print('MAE: {} kcal/mol'.format(test_offset_MAE))

    avg_rst_df = avg_rst_df.round(2)

    return avg_rst_df, test_offset_MAE



def plot_convergence(dataframe, n_calls):

    print('Plotting convergence...')

    # x values
    x = list(range(1, n_calls + 1))

    # y values
    cumltv_MAE = [dataframe.loc[dataframe['Fold'] == fold, 'MAE/MAD'].tolist() for fold in range(1, 6)]
    cumltv_MAE = list(zip(*cumltv_MAE))
    y = [statistics.mean(call) for call in cumltv_MAE]

    # standard devation
    stdev = [statistics.stdev(call) for call in cumltv_MAE]

    # standard devation bounds
    y1 = [i - sd for i, sd in zip(y, stdev)]
    y2 = [i + sd for i, sd in zip(y, stdev)]

    # plot mean line
    plt.figure()
    line = plt.plot(x, y,
                    color='black',
                    linewidth=0.5,
                    label='Average MAE over 5 folds')

    # plot standard deviation bounds
    fill = plt.fill_between(x, y1, y2,
                            fc='lightsteelblue',
                            ec='lightsteelblue',
                            label='Standard deviation')

    plt.xlabel('Number of calls n')
    plt.ylabel('MAE after n calls')

    plt.legend()

    plt.savefig(output_dr + '/convergence_plot.png')
    print('Saved ' + output_dr + '/convergence_plot.png.')


def svr(fold, fold_num):
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
        regr.fit(train_X, train_y)

        predicted_y = regr.predict(validate_X)

        # calculate some statistics on validate set:
        MAE = mean_absolute_error(validate_y, predicted_y)
        MAD_validate = validate_y_df.mad()

        MAEMAD = MAE / MAD_validate
        print('MAE/MAD:', MAEMAD)

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
            startpoint_MAEMAD = MAEMAD
            # keep track of models
            all_models.append([MAEMAD, fold_num, tau, r_value, nested_lst_result])

            # write all model files:
            with open(output_dr + '/fold_' + str(fold_num) + '_SVM_model.svm', 'wb') as file:
                pickle.dump(regr, file)

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
    print(search_result.x)

    print('——————————————————————————————————————————')

    # return skopt object and highest scoring model for this fold:
    return search_result, all_models[-1]


def run_svr(kfolds):

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
        OptimizeResult, top_model = svr(fold, fold_num)

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
        with open(output_dr + '/logfile.txt', 'a') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(['Finished fold', fold_num, 'at', str(time.ctime())])

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
        with open(output_dr + '/fold_' + str(internal_fold_num) + '_internal_validation.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Experimental dGoffset (kcal/mol)', 'Predicted dGoffset (kcal/mol)'])
            for row in internal_validation:
                writer.writerow(row)

    MAEtauR_df = pd.DataFrame(MAEtauR_results_per_fold[1:], columns=MAEtauR_results_per_fold[0])
    cumulative_MAEtauR_df = pd.concat([cumulative_MAEtauR_df, MAEtauR_df])

    # Save to CSV
    cumulative_MAEtauR_df_save_loc = output_dr + '/dGoffset_SVR_MAEtauR_outputs.csv'

    if os.path.exists(cumulative_MAEtauR_df_save_loc):
        os.remove(cumulative_MAEtauR_df_save_loc)
        cumulative_MAEtauR_df.to_csv(path_or_buf=cumulative_MAEtauR_df_save_loc, index=True)
        print('Existing file overwritten.')
    else:
        cumulative_MAEtauR_df.to_csv(path_or_buf=cumulative_MAEtauR_df_save_loc, index=True)

    print('Completed writing cumulative MAE, tau and R to CSV.')

    # Save to CSV
    cumulative_MAE_save_loc = output_dr + '/dGoffset_SVR_BO_MAE.csv'

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

    print('Completed {}-fold cross-validation.'.format(n_splits))
    return kfolds


def separate_train_test(full_dataset):

    # List comprehension for all non-SAMPL4_Guthrie entires.
    train_IDs = [freesolv_df.iloc[i][0]
                 for i in range(len(freesolv_df))
                 if freesolv_df.loc[i, exp_ref_col] != SAMPL4_Guthrie_ref]

    # List comprehension for all SAMPL4_Guthrie entires.
    test_IDs = [freesolv_df.iloc[i][0]
                for i in range(len(freesolv_df))
                if freesolv_df.loc[i, exp_ref_col] == SAMPL4_Guthrie_ref]

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
        descriptors_raw = open(path + 'features_X/mordred_descriptors/all_descriptors.txt', 'r')
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

    return mordred_df


if __name__ == '__main__':
    main()
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Script finished.')
