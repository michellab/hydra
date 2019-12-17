# General:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.model_selection import KFold

# RDKit
from rdkit import Chem
from rdkit import DataStructs

# File path variables
absolute_dGoffset_path = './absolute_dGoffset/'
offset_col_name = 'dGoffset (kcal/mol)'
train_dr = absolute_dGoffset_path + 'train_dr/'
test_dr = absolute_dGoffset_path + 'test_dr/'

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

    # load dfs
    train_df = pd.read_csv(absolute_dGoffset_path + 'train_data.csv', index_col='Unnamed: 0')
    test_df = pd.read_csv(absolute_dGoffset_path + 'test_data.csv')
    test_df_ID = pd.read_csv(absolute_dGoffset_path + 'test_data_index.csv', index_col='Unnamed: 0')

    # test set IDs
    test_ID = test_df_ID.index.tolist()

    # selected external test set data
    target_df = pd.read_csv(absolute_dGoffset_path + 'fp_similarity_target_ligands.csv', index_col='ID')
    target_df = target_df.tail(5)
    target_ID = target_df.index.tolist()
    target_MAE = target_df.iloc[:, 4].tolist()

    # simulate the same 5-fold splitting
    kfolds = split_dataset(train_df, n_splits, random_state)

    # initiate grid plot
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
                cell.set_xlabel('{}\nMAE: {} kcal/mol'.format(target_ID[y], target_MAE[y]),
                                labelpad=10)
            if y == len(target_ID) - 1:
                cell.yaxis.set_label_position('right')
                cell.set_ylabel('Fold {}'.format(x + 1),
                                labelpad=25,
                                rotation=0)

    plt.tight_layout()
    plt.savefig(absolute_dGoffset_path + 'lowest_fingerprint_similarity.png')
    print('Finished plotting figure.')


# if __name__ == '__main__':
#
#     # load dfs
#     train_df = pd.read_csv(absolute_dGoffset_path + 'train_data.csv', index_col='Unnamed: 0')
#     test_df = pd.read_csv(absolute_dGoffset_path + 'test_data.csv')
#     test_df_ID = pd.read_csv(absolute_dGoffset_path + 'test_data_index.csv', index_col='Unnamed: 0')
#
#     # test set IDs
#     test_ID = test_df_ID.index.tolist()
#
#     # selected external test set data
#     target_df = pd.read_csv(absolute_dGoffset_path + 'fp_similarity_target_ligands.csv', index_col='ID')
#
#     best = list(range(5))
#     worst = [-(x + 1) for x in reversed(best)]
#     dict = {'best': best, 'worst': worst}
#
#     plt.figure()
#
#
#
#
#     plt.subplot(2, 1, 1)
#
#     target = target_df.iloc[dict['best'], :]
#     target_ID = target.index.tolist()
#     target_MAE = target.iloc[:, 4].tolist()
#
#     # simulate the same 5-fold splitting
#     kfolds = split_dataset(train_df, n_splits, random_state)
#
#     # initiate grid plot
#     fig1, axs1 = plt.subplots(nrows=n_splits,
#                             ncols=len(target_ID),
#                             facecolor='w',
#                             edgecolor='k',
#                             sharex=True,
#                             sharey=True)
#
#     # # add a big axis and hide frame
#     # fig.add_subplot(111, frameon=False)
#     #
#     # # hide tick and tick label of the big axis
#     # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     # plt.xlabel('Fingerprint similarity')
#     # plt.ylabel('Density')
#
#     # iterate through all selected external test set data
#     for j, mol, MAE in zip(range(len(target_ID)), target_ID, target_MAE):
#
#         # target external test set entry fingerprint generation
#         target_suppl = Chem.SDMolSupplier(test_dr + str(mol) + '.sdf')
#         target_fp = Chem.RDKFingerprint(target_suppl[0])
#
#         # iterate through each fold for selected targeted external test set entry
#         for i, fold in zip(range(n_splits), kfolds):
#
#             fold_num = j + 1
#
#             # retrieve IDs
#             train_IDs = fold[0][0].index.tolist()
#             validate_IDs = fold[0][1].index.tolist()
#
#             # retrieve SDFs
#             train_suppl = [Chem.SDMolSupplier(train_dr + str(sdf) + '.sdf') for sdf in train_IDs]
#             valdtn_suppl = [Chem.SDMolSupplier(train_dr + str(sdf) + '.sdf') for sdf in validate_IDs]
#
#             # generate fingerprints
#             train_fp = [Chem.RDKFingerprint(mol[0]) for mol in train_suppl]
#             valdtn_fp = [Chem.RDKFingerprint(mol[0]) for mol in valdtn_suppl]
#
#             # compute similarities
#             train_similarity = [DataStructs.FingerprintSimilarity(target_fp, train_mol) for train_mol in train_fp]
#             valdnt_similarity = [DataStructs.FingerprintSimilarity(target_fp, valdnt_mol) for valdnt_mol in valdtn_fp]
#
#             # plot densities
#             sns.distplot(train_similarity,
#                          hist=False,
#                          kde=True,
#                          kde_kws={'linewidth': 2},
#                          label='Train similarity',
#                          ax=axs1[i, j])
#
#             sns.distplot(valdnt_similarity,
#                          hist=False,
#                          kde=True,
#                          kde_kws={'linewidth': 2},
#                          label='Validation similarity',
#                          ax=axs1[i, j])
#
#             # remove all subplot legends and add global legend
#             axs1[i, j].get_legend().remove()
#             handles, labels = axs1[i, j].get_legend_handles_labels()
#             # fig.legend(handles, labels, loc='lower center')
#             fig1.legend(handles, labels,
#                        loc='lower center',
#                        bbox_to_anchor=(0.5, 0.0),
#                        ncol=2)
#
#     # # add row and column labels
#     # for x, row in enumerate(axs):
#     #     for y, cell in enumerate(row):
#     #         if x == 0:
#     #             cell.xaxis.set_label_position('top')
#     #             cell.set_xlabel('{}\nMAE: {} kcal/mol'.format(target_ID[y], target_MAE[y]),
#     #                             labelpad=10)
#     #         if y == len(target_ID) - 1:
#     #             cell.yaxis.set_label_position('right')
#     #             cell.set_ylabel('Fold {}'.format(x + 1),
#     #                             labelpad=25,
#     #                             rotation=0)
#
#
#
#
#     plt.subplot(2, 1, 2)
#
#     target = target_df.iloc[dict['best'], :]
#     target_ID = target.index.tolist()
#     target_MAE = target.iloc[:, 4].tolist()
#
#     # simulate the same 5-fold splitting
#     kfolds = split_dataset(train_df, n_splits, random_state)
#
#     # initiate grid plot
#     fig2, axs2 = plt.subplots(nrows=n_splits,
#                             ncols=len(target_ID),
#                             facecolor='w',
#                             edgecolor='k',
#                             sharex=True,
#                             sharey=True)
#
#     # # add a big axis and hide frame
#     # fig.add_subplot(111, frameon=False)
#     #
#     # # hide tick and tick label of the big axis
#     # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     # plt.xlabel('Fingerprint similarity')
#     # plt.ylabel('Density')
#
#     # iterate through all selected external test set data
#     for j, mol, MAE in zip(range(len(target_ID)), target_ID, target_MAE):
#
#         # target external test set entry fingerprint generation
#         target_suppl = Chem.SDMolSupplier(test_dr + str(mol) + '.sdf')
#         target_fp = Chem.RDKFingerprint(target_suppl[0])
#
#         # iterate through each fold for selected targeted external test set entry
#         for i, fold in zip(range(n_splits), kfolds):
#
#             fold_num = j + 1
#
#             # retrieve IDs
#             train_IDs = fold[0][0].index.tolist()
#             validate_IDs = fold[0][1].index.tolist()
#
#             # retrieve SDFs
#             train_suppl = [Chem.SDMolSupplier(train_dr + str(sdf) + '.sdf') for sdf in train_IDs]
#             valdtn_suppl = [Chem.SDMolSupplier(train_dr + str(sdf) + '.sdf') for sdf in validate_IDs]
#
#             # generate fingerprints
#             train_fp = [Chem.RDKFingerprint(mol[0]) for mol in train_suppl]
#             valdtn_fp = [Chem.RDKFingerprint(mol[0]) for mol in valdtn_suppl]
#
#             # compute similarities
#             train_similarity = [DataStructs.FingerprintSimilarity(target_fp, train_mol) for train_mol in train_fp]
#             valdnt_similarity = [DataStructs.FingerprintSimilarity(target_fp, valdnt_mol) for valdnt_mol in valdtn_fp]
#
#             # plot densities
#             sns.distplot(train_similarity,
#                          hist=False,
#                          kde=True,
#                          kde_kws={'linewidth': 2},
#                          label='Train similarity',
#                          ax=axs2[i, j])
#
#             sns.distplot(valdnt_similarity,
#                          hist=False,
#                          kde=True,
#                          kde_kws={'linewidth': 2},
#                          label='Validation similarity',
#                          ax=axs2[i, j])
#
#             # remove all subplot legends and add global legend
#             axs2[i, j].get_legend().remove()
#             handles, labels = axs2[i, j].get_legend_handles_labels()
#             # fig.legend(handles, labels, loc='lower center')
#             fig2.legend(handles, labels,
#                        loc='lower center',
#                        bbox_to_anchor=(0.5, 0.0),
#                        ncol=2)
#
#     # # add row and column labels
#     # for x, row in enumerate(axs):
#     #     for y, cell in enumerate(row):
#     #         if x == 0:
#     #             cell.xaxis.set_label_position('top')
#     #             cell.set_xlabel('{}\nMAE: {} kcal/mol'.format(target_ID[y], target_MAE[y]),
#     #                             labelpad=10)
#     #         if y == len(target_ID) - 1:
#     #             cell.yaxis.set_label_position('right')
#     #             cell.set_ylabel('Fold {}'.format(x + 1),
#     #                             labelpad=25,
#     #                             rotation=0)
#
#
#
#
#     plt.tight_layout()
#
#
#
#
#
#     plt.savefig(absolute_dGoffset_path + 'test.png')
#
#     print('Finished plotting figure.')
