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
import pprint
import json

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

# statistics
from scipy.stats import shapiro, normaltest, anderson

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
offset_col_name = 'ddGoffset (kcal/mol)'
# SAMPl4_Guthrie experimental reference in FreeSolv.
SAMPL4_Guthrie_ref = 'SAMPL4_Guthrie'
# Experimental reference column name.
exp_ref_col = 'experimental reference (original or paper this value was taken from)'

model_type = 'SVM'

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

    # initiate log file
    logging.basicConfig(filename= output_dr + 'testing_logfile.txt',
                        filemode='a',
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)

    logging.info('Starting ddGhydr_test_{}.py.'.format(model_type))

    # load DataFrames from training script
    dtrain_df = pd.read_csv(path + 'dtrain_data.csv', index_col='Unnamed: 0')
    dtest_df = pd.read_csv(path + 'dtest_data.csv', index_col='Unnamed: 0')
    cumulative_MAE_df = pd.read_csv(output_dr + 'ddGoffset_' + model_type +  '_BO_MAE.csv', index_col='Unnamed: 0')

    # load in kfolds nested list from training
    with open(path + 'kfolds.json', "rb") as f:
        kfolds = pickle.load(f)

    # testing
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Testing...')
    logging.info('Started testing....')
    predicted_offset, predicted_offset_mae = predict_offset(dtest_df)
    corrected_hydr, corrected_hydr_mae, mobley_hydr_mae = correct_hydration(predicted_offset)
    print('Finished external testing.')
    logging.info('Testing complete.')

    # plot graphs
    print('Plotting figures...')
    plot_convergence(cumulative_MAE_df, 40)

    plot_scatter(predicted_offset,
                 [1, 'Experimental ddGoffset (kcal/mol)'],
                 [2, 'Averaged predicted ddGoffset (kcal/mol)'],
                 title='External test set ddGoffsets',
                 MAE=predicted_offset_mae)

    plot_scatter(corrected_hydr,
                 [1, 'Experimental ddGhydr (kcal/mol)'],
                 [2, 'Calculated ddGhydr (kcal/mol)'],
                 title='Original Mobley calculated ddGhydr',
                 MAE=mobley_hydr_mae)

    plot_scatter(corrected_hydr,
                 [1, 'Experimental ddGhydr (kcal/mol)'],
                 [4, 'Corrected calculated ddGhydr (kcal/mol)'],
                 title='External test set corrected calculated ddGhydr',
                 MAE=corrected_hydr_mae)

    # # fingerprint similarity
    # print('Plotting fingerprint similarity plots...')
    # plot_fingerprint_similarity(corrected_hydr, dtrain_df, dtest_df, kfolds, 'highest')
    # plot_fingerprint_similarity(corrected_hydr, dtrain_df, dtest_df, kfolds, 'lowest')
    print('Finished plotting figures.')

    logging.info('Finished ddGhydr_test_{}.py.'.format(model_type))

    ####################### set two main report colour themes and one extra #######################

    ####################### FG counter to be added to end of plot_fingerprint_similarity #######################


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

    # retrieve best and worst predicted ddGhydr
    corr_AE_sort_df = corrected_hydr.sort_values(by='Corrected calculated ddGhydr absolute error (kcal/mol)')
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

    # relative experimental ddGhydr
    test_exp = test_fs_df.iloc[:, 3].tolist()
    dtest_exp = [col - row for col in test_exp for row in test_exp if (col - row) != 0]

    # list index where value was zero
    diagonal = [col - row for col in test_exp for row in test_exp]
    diagonal_index = [i for i, x in enumerate(diagonal) if x == 0]

    # calculated ddGhydr
    test_calc = test_fs_df.iloc[:, 5].tolist()
    dtest_calc = [col - row for col in test_calc for row in test_calc]
    dtest_calc = pd.DataFrame(dtest_calc)
    dtest_calc = dtest_calc.drop(diagonal_index)
    dtest_calc = dtest_calc.values.tolist()
    dtest_calc = [x[0] for x in dtest_calc]

    # # calculated ddGhydr uncertainty
    # dtest_calc_err = test_fs_df.iloc[:, 6].tolist()

    # corrected calculated ddGhydr using predicted ddGoffsets
    davg_offsets = predicted_offset['Averaged predicted ddGoffset (kcal/mol)']
    dcorr_calc = [calc + err for calc, err in zip(dtest_calc, davg_offsets)]

    # calculated ddGhydr absolute error
    dcalc_AE = [abs(exp - calc) for exp, calc in zip(dtest_exp, dtest_calc)]

    # corrected calculated ddGhydr propogated absolute error
    # corr_AE = (err1**2 + err2**2)**0.5
    dcorr_AE = [abs(exp - calc) for exp, calc in zip(dtest_exp, dcorr_calc)]

    # create df
    dcorr_dict = {'ID': predicted_offset['ID'].tolist(),
                  'Experimental ddGhydr (kcal/mol)': dtest_exp,
                  'Calculated ddGhydr (kcal/mol)': dtest_calc,
                  'Calculated ddGhydr absolute error (kcal/mol)': dcalc_AE,
                  'Corrected calculated ddGhydr (kcal/mol)': dcorr_calc,
                  'Corrected calculated ddGhydr absolute error (kcal/mol)': dcorr_AE}

    dcorr_df = pd.DataFrame(dcorr_dict).round(2)

    # calculate MAEs
    dcalc_MAE = statistics.mean(dcalc_AE)
    print('Mobley calculated MAE: {} kcal/mol'.format(round(dcalc_MAE, 2)))
    dcorr_MAE = statistics.mean(dcorr_AE)
    print('Corrected calculated MAE: {} kcal/mol'.format(round(dcorr_MAE, 2)))

    return dcorr_df, dcorr_MAE, dcalc_MAE


def calc_mae(dataframe, model):

    model_df = dataframe.loc[dataframe['Model number'] == model]
    abs_err = model_df['Absolute error (kcal/mol)'].tolist()
    MAE = statistics.mean(abs_err)

    return MAE


def svr_predict(model_num, test_entry):

    with open(output_dr + 'fold_' + str(model_num) + '_SVM_model.svm', 'rb') as file:
        model = pickle.load(file)

    return model.predict(test_entry)


def predict_offset(test_set):

    print('Predicting offsets...')

    # load in testing set
    dtest_ID = test_set.index
    dtest_X = test_set.drop(columns='ddGoffset (kcal/mol)').values
    dtest_y = test_set['ddGoffset (kcal/mol)'].values

    # empty df for external testing results
    dtest_rst = pd.DataFrame()

    # peform prediction using each model
    num_models = list(range(1, n_splits + 1))
    for model in num_models:
        # call SVR prediction function
        dsvr_rst = svr_predict(model, dtest_X)

        # write results per fold into dictionary and load into df
        model_rst = {'ID': dtest_ID,
                     'Model number': [model for i in range(len(dtest_ID))],
                     'Experimental ddGoffset (kcal/mol': dtest_y,
                     'Predcted ddGoffset (kcal/mol)': dsvr_rst,
                     'Absolute error (kcal/mol)': abs(dtest_y - dsvr_rst)}

        dtest_rst = pd.concat([dtest_rst, pd.DataFrame(model_rst)])

    # calculate MAE values
    MAE_lst = [calc_mae(dtest_rst, model) for model in num_models]
    print('MAE values between experimental and predicted ddGoffset values:\n')
    for model, model_MAE in enumerate(MAE_lst):
        print('Model {} MAE: {} kcal/mol'.format(model + 1, round(model_MAE, 2)))
    print('\nAverage MAE: {} kcal/mol'.format(round(statistics.mean(MAE_lst), 2)))

    # round whole results dataframe to two decimal places
    dtest_rst = dtest_rst.round(2)

    # average predicted offset values
    dprdt_offsets = [dtest_rst.loc[dtest_rst['Model number'] == model, 'Predcted ddGoffset (kcal/mol)'].tolist()
                     for model in num_models]
    dprdt_offsets = list(zip(*dprdt_offsets))
    davg_offsets = [statistics.mean(offset_set) for offset_set in dprdt_offsets]

    # write results to df
    davg_rst = {}

    davg_rst['ID'] = dtest_ID
    davg_rst['Experimental ddGoffset (kcal/mol)'] = dtest_y
    davg_rst['Averaged predicted ddGoffset (kcal/mol)'] = davg_offsets
    davg_rst['Absolute error (kcal/mol)'] = abs(dtest_y - davg_offsets)

    davg_rst_df = pd.DataFrame(davg_rst)

    # MAE
    print('MAE between experimental and averaged predicted ddGoffsets:')
    test_offset_MAE = round(statistics.mean(abs(dtest_y - davg_offsets)), 2)
    print('MAE: {} kcal/mol'.format(test_offset_MAE))

    davg_rst_df = davg_rst_df.round(2)

    return davg_rst_df, test_offset_MAE


if __name__ == '__main__':

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Script started on {}'.format(time.ctime()))

    main()

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Script finished on {}.'.format(time.ctime()))
