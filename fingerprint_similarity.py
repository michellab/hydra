# General:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# SVM:
from sklearn.model_selection import KFold

# RDKit
from rdkit import Chem
from rdkit import DataStructs

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

absolute_dGoffset_path = './absolute_dGoffset/'

offset_col_name = 'dGoffset (kcal/mol)'

# set data processing configurations:
PCA_threshold = 0.95  # Keeps n dimensions for x variance explained
replicates = 30  # Number of replicates per subject model
n_calls = 40  # Number of Bayesian optimisation loops for hyperparameter optimisation, 40 is best for convergence, > 60 scales to very expensive
startpoint_BO = np.inf  # Point to consider top-performing model from (MAE/MAD); 1.0 = no improvement on test-set variance
ensemble_size = 10  # Amount of top-scoring models to retain per fold-dataset combination
# KFold parameters:
n_splits = 5  # Number of K-fold splits
random_state = 2  # Random number seed


def split_dataset(dataset, n_splits, random_state):
    """KFold implementation for pandas DataFrame.
    (https://stackoverflow.com/questions/45115964/separate-pandas-dataframe-using-sklearns-kfold)"""

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

    return kfolds

if __name__ == '__main__':

    train_df_save_loc = absolute_dGoffset_path + 'train_data.csv'
    train_df = pd.read_csv(train_df_save_loc, index_col='Unnamed: 0')
    train_dr = absolute_dGoffset_path + 'train_dr/'

    test_df_save_loc = absolute_dGoffset_path + 'test_data.csv'
    test_dr = absolute_dGoffset_path + 'test_dr/'
    test_df = pd.read_csv(test_df_save_loc)

    test_df_index_save_loc = absolute_dGoffset_path + 'test_data_index.csv'
    test_df_ID = pd.read_csv(test_df_index_save_loc, index_col='Unnamed: 0')
    test_ID = test_df_ID.index.tolist()

    worst_best_df = pd.read_csv(absolute_dGoffset_path + 'worst_best_ligands.csv', index_col='ID')
    worst_best_ID = worst_best_df.index.tolist()
    worst_best_MAE = worst_best_df.iloc[:, 4].tolist()

    kfolds = split_dataset(train_df, n_splits, random_state)

    for mol, MAE in zip(worst_best_ID, worst_best_MAE):

        trgt_suppl = Chem.SDMolSupplier(test_dr + str(mol) + '.sdf')
        trgt_fp = Chem.RDKFingerprint(trgt_suppl[0])

        ID = mol

        fig, axs = plt.subplots(1, 5,
                                figsize=(15, 6),
                                facecolor='w',
                                edgecolor='k',
                                sharex=True,
                                sharey=True)
        # fig.subplots_adjust(hspace=.5, wspace=.001)

        fig.suptitle(ID + ' training and validation fingerprint similarity densities\nMAE: '
                     + str(MAE) + ' (kcal/mol)')

        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel('Fingerprint similarity')
        plt.ylabel('Density')

        fold_num = 1
        for i, fold in zip(range(5), kfolds):
            train_IDs = fold[0][0].index.tolist()
            validate_IDs = fold[0][1].index.tolist()

            # retrieve SDFs
            train_suppl = [Chem.SDMolSupplier(train_dr + str(sdf) + '.sdf')
                           for sdf in train_IDs]

            valdtn_suppl = [Chem.SDMolSupplier(train_dr + str(sdf) + '.sdf')
                            for sdf in validate_IDs]

            # generate fingerprints
            train_fp = [Chem.RDKFingerprint(mol[0]) for mol in train_suppl]
            valdtn_fp = [Chem.RDKFingerprint(mol[0]) for mol in valdtn_suppl]

            # compute similarities
            train_similarity = [DataStructs.FingerprintSimilarity(trgt_fp, train_mol)
                                for train_mol in train_fp]

            valdnt_similarity = [DataStructs.FingerprintSimilarity(trgt_fp, valdnt_mol)
                                for valdnt_mol in valdtn_fp]

            # density plot
            sns.distplot(train_similarity,
                         hist=False,
                         kde=True,
                         kde_kws={'linewidth': 3},
                         label='Train similarity',
                         ax=axs[i])

            sns.distplot(valdnt_similarity,
                         hist=False,
                         kde=True,
                         kde_kws={'linewidth': 3},
                         label='Validation similarity',
                         ax=axs[i])

            ############# uncomment when grid is used #############
            #
            # for i, row in enumerate(axs):
            #     for j, cell in enumerate(row):
            #         cell.imshow(np.random.rand(32, 32))
            #         if i == len(axs) - 1:
            #             cell.set_xlabel("noise column: {0:d}".format(j + 1))
            #         if j == 0:
            #             cell.set_ylabel("noise row: {0:d}".format(i + 1))
            #
            # plt.tight_layout()

            axs[i].set_title('Fold ' + str(fold_num))
            axs[i].get_legend().remove()
            handles, labels = axs[i].get_legend_handles_labels()
            fig.legend(handles, labels, loc='right')
            save_loc = absolute_dGoffset_path + ID + '_fold_' + str(fold_num) + '_fp_hist.png'
            plt.savefig(save_loc)

            fold_num += 1

