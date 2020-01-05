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
SDF_dr = '../datasets/sdffiles/'
freesolv_loc = '../datasets/database.txt'
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
    logging.basicConfig(filename= output_dr + 'testing_logfile.txt',
                        filemode='a',
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)
    logging.info('Starting dGhydr_testing_{}.py.'.format(model_type))

    # load DataFrames from training script
    train_df = pd.read_csv(path + 'train_data.csv', index_col='Unnamed: 0')
    test_df = pd.read_csv(path + 'test_data.csv', index_col='Unnamed: 0')
    cumulative_MAE_df = pd.read_csv(output_dr + 'dGoffset_' + model_type +  '_BO_MAE.csv', index_col='Unnamed: 0')

    # load in kfolds nested list from training
    with open(path + 'kfolds.json', "rb") as jsonfile:
        kfolds = json.load(jsonfile)

    # testing
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Testing...')
    logging.info('Testing started...')
    predicted_offset, predicted_offset_mae = predict_offset(test_df)
    corrected_hydr, corrected_hydr_mae, mobley_hydr_mae = correct_hydration(predicted_offset)
    print('Finished external testing.')
    logging.info('Testing complete.')

    # plot graphs
    print('Plotting figures...')
    plot_convergence(cumulative_MAE_df, 40)

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
    print('Finished plotting figures.')

    logging.info('Finished dGhydr_testing_{}.py.'.format(model_type))


def draw_structure_panel(sdf_suppl, legend, filename):
    """Draw RDKit.Draw in panel format.

    sdf_suppl: list of SDF pathnames.
    legend: list of strings to append as legends for each SDF.
    filename: filename in string format without a '/'.

    Returns: PNG file saved at filename location."""

    suppl = [SDMolSupplier(sdf) for sdf in sdf_suppl]
    mols = [x[0] for x in suppl if x is not None]
    for mol in mols:
        tmp = AllChem.Compute2DCoords(mol)

    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200), legends=legend)
    img.save(filename)
    print('Finished drawing structure panel {}.'.format(filename))


def plot_fingerprint_similarity(corrected_hydr, train_df, test_df, kfolds, mode):

    # retrieve best and worst predicted dGhydr
    corr_AE_sort_df = corrected_hydr.sort_values(by='Corrected calculated dGhydr absolute error (kcal/mol)')
    target_df = pd.concat([corr_AE_sort_df.head(), corr_AE_sort_df.tail()])
    target_df = target_df.set_index('ID')

    test_ID = test_df.index.tolist()

    # selected external test set data
    if mode == 'highest':
        target_df = target_df.tail(5)
    elif mode == 'lowest':
        target_df = target_df.head(5)

    target_ID = target_df.index.tolist()
    target_MAE = target_df.iloc[:, 4].tolist()

    # initiate grid plot
    plt.figure()
    fig, axs = plt.subplots(nrows=n_splits,
                            ncols=len(target_ID),
                            figsize=(15, 10),  # 15, 6
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

    # statistics dictionary per ligand
    stats = {'mean': [], 'percentile': []}

    # iterate through all selected external test set data
    for j, mol, MAE in zip(range(len(target_ID)), target_ID, target_MAE):

        # target external test set entry fingerprint generation
        target_suppl = Chem.SDMolSupplier(test_dr + str(mol) + '.sdf')
        target_fp = Chem.RDKFingerprint(target_suppl[0])

        # statistics dictionary per fold
        mol_stats = {'mean': [], 'percentile': []}

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

            # means
            train_mean = statistics.mean(train_similarity)
            valdnt_mean = statistics.mean(valdnt_similarity)
            mol_stats['mean'].append(statistics.mean([train_mean, valdnt_mean]))

            # 95th percentile
            train_percentile = np.percentile(train_similarity, 95)
            valdnt_percentile = np.percentile(valdnt_similarity, 95)
            mol_stats['percentile'].append(statistics.mean([train_percentile, valdnt_percentile]))

        # append averaged statistics per fold
        stats['mean'].append(round(statistics.mean(mol_stats['mean']), 2))
        stats['percentile'].append(round(statistics.mean(mol_stats['percentile']), 2))

    # add row and column labels
    for x, row in enumerate(axs):
        for y, cell in enumerate(row):
            if x == 0:
                cell.xaxis.set_label_position('top')
                cell.set_xlabel('{}\nMAE = {} kcal/mol'.format(target_ID[y], round(target_MAE[y], 2)),
                                labelpad=10)
            if x == len(kfolds) - 1:
                cell.xaxis.set_label_position('bottom')
                cell.set_xlabel(
                    'Mean = {} kcal/mol\n95th percentile = {} kcal/mol'.format(stats['mean'][y], stats['percentile'][y]),
                    labelpad=30)
            if y == len(target_ID) - 1:
                cell.yaxis.set_label_position('right')
                cell.set_ylabel('Fold {}'.format(x + 1),
                                labelpad=25,
                                rotation=0)

    plt.tight_layout()

    filename = figures_dr + mode + '_fingerprint_similarity.png'
    plt.savefig(filename)
    print('Finished plotting figure {}.'.format(filename))

    # draw respective chemical structures
    draw_structure_panel(sdf_suppl=[test_dr + str(mol) + '.sdf' for mol in target_ID],
                         legend=['MAE: {} kcal/mol'.format(round(mae, 2)) for mae in target_MAE],
                         filename=figures_dr + mode + '_chemical_structures.png')
    print('Finished drawing chemical structures.')


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

    filename = figures_dr + str(title).lower().replace(' ', '_') + '.png'
    plt.savefig(filename)
    print('Saved', filename)


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

    filename = figures_dr + 'convergence_plot.png'
    plt.savefig(filename)
    print('Saved ', filename)


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


def calc_mae(dataframe, model):

    model_df = dataframe.loc[dataframe['Model number'] == model]
    abs_err = model_df['Absolute error (kcal/mol)'].tolist()
    MAE = statistics.mean(abs_err)

    return MAE


def svr_predict(model_num, test_entry):

    model = tf.keras.models.load_model(output_dr + 'fold_' + str(model_num) + '_' + model_type + '_model.h5')

    # with open(output_dr + 'fold_' + str(model_num) + '_' + model_type + '_model.' + model_type.lower(), 'rb') as file:
    #     model = pickle.load(file)

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


if __name__ == '__main__':

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Script started on {}'.format(time.ctime()))

    main()

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Script finished.')
