# General:
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle

# Statistics:
import statistics
from sklearn.metrics import r2_score

# RDKit:
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import SDMolSupplier
from rdkit import DataStructs

# Path  variables
path = './'
datasets_dr = '../datasets/'
SDF_dr = datasets_dr + 'sdffiles/'
freesolv_loc = datasets_dr + 'freesolv_database.txt'
train_dr = datasets_dr + 'train_dr/'
test_dr = datasets_dr + 'test_dr/'
output_dr = path + 'output/'
figures_dr = path + 'figures/'

# Global variables:
model_type = 'SVM'
n_calls = 40  # Number of Bayesian optimisation loops for hyperparameter optimisation, 40 is best for convergence, > 60 scales to very expensive
n_splits = 5  # Number of K-fold splits

# Load in FreeSolve
freesolv_df = pd.read_csv(freesolv_loc, sep='; ', engine='python')


def main():

    # initiate log file
    logging.basicConfig(filename= output_dr + 'test_logfile.txt',
                        filemode='a',
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)

    logging.info('Starting dGhydr_{}.py.'.format(model_type))

    # Load in datasets.
    train_df = pd.read_csv(datasets_dr + 'train_data.csv', index_col='Unnamed: 0')
    test_df = pd.read_csv(datasets_dr + 'test_data.csv', index_col='Unnamed: 0')
    cumulative_MAE_df = pd.read_csv(output_dr + 'dGoffset_' + model_type + '_BO_MAE.csv', index_col='Unnamed: 0')
    with open(path + 'kfolds.json', "rb") as jsonfile: kfolds = pickle.load(jsonfile)

    # testing
    logging.info('Started testing....')
    predicted_offset, predicted_offset_mae = predict_offset(test_df)
    corrected_hydr, corrected_hydr_mae, mobley_hydr_mae = correct_hydration(predicted_offset)
    logging.info('Testing complete.')

    # plot graphs
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
    plot_fingerprint_similarity(corrected_hydr, train_df, test_df, kfolds, 'highest')
    plot_fingerprint_similarity(corrected_hydr, train_df, test_df, kfolds, 'lowest')

    logging.info('Finished dGhydr_{}.py.'.format(model_type))


def new_plot():

    corrections = [correction for correction in df["Correction"].values]
    fep_values = [value for value in df["ddG (FEP) (kcal/ mol)"].values]
    exp_values = [value for value in df["ddG (EXP) (kcal/ mol)"].values]
    hybrid_values = [fep - corr for fep, corr in zip(fep_values, corrections)]
    positive_bound = max(list(fep_values + hybrid_values + exp_values))
    negative_bound = min(list(fep_values + hybrid_values + exp_values))
    plt.xlim(negative_bound - 0.1, positive_bound + 0.1)
    plt.ylim(negative_bound - 0.1, positive_bound + 0.1)
    for correction, fep, exp in zip(corrections, fep_values, exp_values):
        fep_corrected = fep - correction
        fep_offset = exp - fep
        fep_corrected_offset = exp - fep_corrected
        if abs(fep_corrected_offset) <= abs(fep_offset):
            line_color = "green"
        else:
            line_color = "red"
        if abs(correction) >= 0.1:
            plt.annotate("", xytext=(fep, exp), xy=(fep_corrected, exp),
                         arrowprops=dict(arrowstyle="->", color=line_color)
                         )
    plt.show()


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
    logging.info('Finished plotting figure {}.'.format(filename))

    # draw respective chemical structures
    draw_structure_panel(sdf_suppl=[test_dr + str(mol) + '.sdf' for mol in target_ID],
                         legend=['MAE: {} kcal/mol'.format(round(mae, 2)) for mae in target_MAE],
                         filename=figures_dr + mode + '_chemical_structures.png')
    logging.info('Finished drawing chemical structures.')


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

    filename = figures_dr + str(title).lower().replace(' ', '_') + '.png'
    plt.savefig(filename)


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
    plt.plot(x, y,
                    color='black',
                    linewidth=0.5,
                    label='Average MAE over 5 folds')

    # plot standard deviation fill bounds
    plt.fill_between(x, y1, y2,
                            fc='lightsteelblue',
                            ec='lightsteelblue',
                            label='Standard deviation')

    plt.xlabel('Number of calls n')
    plt.ylabel('MAE after n calls')

    plt.legend()

    filename = figures_dr + 'convergence_plot.png'
    plt.savefig(filename)


def calc_mae(dataframe, model):

    model_df = dataframe.loc[dataframe['Model number'] == model]
    abs_err = model_df['Absolute error (kcal/mol)'].tolist()
    MAE = statistics.mean(abs_err)

    return MAE


def model_predict(model_num, test_entry):

    with open(output_dr + 'fold_' + str(model_num) + '_' + model_type + '_model.pickle', 'rb') as f:
        model = pickle.load(f)

    return model.predict(test_entry)


def predict_offset(test_set):

    logging.info('Predicting offsets...')

    # load in testing set
    test_id = test_set.index.tolist()
    test_X = test_set.drop(columns='dGoffset (kcal/mol)').values
    test_y = test_set['dGoffset (kcal/mol)'].values

    # empty df for external testing results
    test_rst = pd.DataFrame()

    # perform prediction using each model
    num_models = list(range(1, n_splits + 1))
    for model in num_models:
        # call model prediction function
        model_rst = model_predict(model, test_X)

        # write results per fold into dictionary and load into df
        rst_dict = {'ID': test_id,
                    'Model number': [model for i in range(41)],
                    'Experimental dGoffset (kcal/mol)': test_y,
                    'Predicted dGoffset (kcal/mol)': model_rst,
                    'Absolute error (kcal/mol)': [float(abs(x-y)) for x, y in zip(test_y, model_rst)]}

        test_rst = pd.concat([test_rst, pd.DataFrame(rst_dict)])

    # calculate MAE values
    MAE_lst = [calc_mae(test_rst, model) for model in num_models]
    print('MAE values between experimental and predicted dGoffset values:\n')
    for model, model_MAE in enumerate(MAE_lst): print('Model {} MAE: {} kcal/mol'.format(model + 1, round(model_MAE, 2)))
    print('\nAverage MAE: {} kcal/mol'.format(round(statistics.mean(MAE_lst), 2)))

    # # round whole results dataframe to two decimal places
    # test_rst = test_rst.round(2)

    # average predicted offset values
    prdt_offsets = [test_rst.loc[test_rst['Model number'] == model, 'Predicted dGoffset (kcal/mol)'].tolist()
                    for model in num_models]
    prdt_offsets = list(zip(*prdt_offsets))
    avg_offsets = [statistics.mean(offset_set) for offset_set in prdt_offsets]

    # write results to df
    avg_rst = {'ID': test_id,
               'Experimental dGoffset (kcal/mol)': test_y,
               'Averaged predicted dGoffset (kcal/mol)': avg_offsets,
               'Absolute error (kcal/mol)': abs(test_y - avg_offsets)}

    avg_rst_df = pd.DataFrame(avg_rst)

    print('\ndGoffset df\n')
    print(avg_rst_df)

    # MAE
    print('MAE between experimental and averaged predicted dGoffsets:')
    test_offset_MAE = round(statistics.mean(abs(test_y - avg_offsets)), 2)
    print('MAE: {} kcal/mol'.format(test_offset_MAE))

    avg_rst_df = avg_rst_df.round(2)

    return avg_rst_df, test_offset_MAE


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

    main()

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Script finished on {}.'.format(time.ctime()))
