import pandas as pd
import os
import numpy as np
import shutil
import time
from tqdm import tqdm

# Path variables:
path = './'
datasets_dr = '../'
SDF_dr = datasets_dr + 'sdffiles/'
freesolv_loc = datasets_dr + 'freesolv_database.txt'
train_dr = datasets_dr + 'train_dr/'
test_dr = datasets_dr + 'test_dr/'
for dr in [train_dr, test_dr]:
    if os.path.isdir(dr):
        shutil.rmtree(dr)
        print('Existing directory {} overwritten.'.format(dr))
        os.mkdir(dr)
    else:
        os.mkdir(dr)

# Load in FreeSolve.
freesolv_df = pd.read_csv(freesolv_loc, sep='; ', engine='python')
# SAMPl4_Guthrie experimental reference in FreeSolv.
SAMPL4_Guthrie_ref = 'SAMPL4_Guthrie'
# Experimental reference column name.
exp_ref_col = 'experimental reference (original or paper this value was taken from)'


def main():

    # Load in features, labels and null labels sorted by index
    reduced_X = pd.read_csv(datasets_dr + 'features_X/reduced_features.csv', index_col='ID').sort_index()
    true_y = pd.read_csv(datasets_dr + 'labels_y/experimental_labels.csv', index_col='ID').sort_index()
    null_y = pd.read_csv(datasets_dr + 'labels_y/null_experimental_labels.csv', index_col='ID').sort_index()

    # Split features, labels and null labels into training and external testing sets
    train_X, test_X = split_train_test(reduced_X)
    train_y, test_y = split_train_test(true_y)
    train_null, test_null = split_train_test(null_y)

    # Write absolute data sets to CSV
    create_absolute_train_test(train_X, train_y, 'train')
    create_absolute_train_test(test_X, test_y, 'test')
    create_absolute_train_test(train_X, train_null, 'null_train')
    create_absolute_train_test(test_X, test_null, 'null_test')

    # Compile full and null complete data sets
    full_dataset = pd.concat([reduced_X, true_y], axis=1, sort=False)
    null_dataset = pd.concat([reduced_X, null_y], axis=1, sort=False)

    # Split compiled complete data sets into trianing and external testing sets.
    train_full, test_full = split_train_test(full_dataset)
    train_full_null, test_full_null = split_train_test(null_dataset)

    # Write relative data sets to CSV
    create_relative_train_test(train_full, 'dtrain')
    create_relative_train_test(test_full, 'dtest')
    create_relative_train_test(train_full_null, 'null_dtrain')
    create_relative_train_test(test_full_null, 'null_dtest')


def sum_error(error1, error2):
    """Returns sum propagated error between two values."""
    return (error1 ** 2 + error2 ** 2) ** 0.5


def create_relative_train_test(dataframe, set_type):
    """col - row, where col and row notation is taken from matrices nomenclature.
    Index values must be Mobley IDs.
    set_type: 'train' or 'test'
    matrix diagonal is not omitted
    note: label column will still be called 'dGoffset (kcal/mol)' and not 'ddGoffset (kcal/mol)'"""

    # filename to be written
    csv = datasets_dr + set_type + '_data.csv'
    if os.path.exists(csv): os.remove(csv)

    for id1, col in tqdm(dataframe.iterrows(), total=dataframe.shape[0],
                         desc='Writing {}_data.csv'.format(set_type), unit_scale=True, leave=True, ncols=100):

        for id2, row in dataframe.iterrows():
            df = pd.concat(
                [pd.DataFrame([col[0:-1] - row[0:-1]], index=[str(id1) + '~' + str(id2)]),
                 pd.DataFrame({'uncertainty (kcal/mol)': sum_error(col[-1], row[-1])}, index=[str(id1)+'~'+str(id2)])],
                axis=1
            )
            if not df.iloc[0, 0] == 0:
                with open(csv, 'a') as file:
                    df.to_csv(path_or_buf=file, mode='a', index=True, index_label='ID', header=file.tell() == 0)


def create_absolute_train_test(features, labels, set_type):

    # filename to be written
    csv = datasets_dr + set_type + '_data.csv'
    if os.path.exists(csv): os.remove(csv)

    # iterate through rows
    for (id1, X), (id2, y) in tqdm(zip(features.iterrows(), labels.iterrows()), total=features.shape[0],
                         desc='Writing {}_data.csv'.format(set_type), unit_scale=True, leave=True, ncols=100):

        # write row to CSV file
        row = pd.concat([pd.DataFrame([X]), pd.DataFrame([y])], axis=1)
        with open(csv, 'a') as file:
            row.to_csv(path_or_buf=file, mode='a', index=True, index_label='ID', header=file.tell() == 0)

        # copy SDF files
        sdf = str(id1) + '.sdf'
        if set == 'train':
            shutil.copyfile(SDF_dr + sdf, train_dr + sdf)
        elif set == 'test':
            shutil.copyfile(SDF_dr + sdf, test_dr + sdf)


def split_train_test(dataframe):

    # List comprehension for all non-SAMPL4_Guthrie entries.
    train_ids = [freesolv_df.iloc[i][0]
                 for i in range(len(freesolv_df))
                 if freesolv_df.loc[i, exp_ref_col] != SAMPL4_Guthrie_ref]

    # List comprehension for all SAMPL4_Guthrie entries.
    test_ids = [freesolv_df.iloc[i][0]
                for i in range(len(freesolv_df))
                if freesolv_df.loc[i, exp_ref_col] == SAMPL4_Guthrie_ref]

    train_df = dataframe.drop(test_ids)
    test_df = dataframe.drop(train_ids)

    return train_df, test_df


if __name__ == '__main__':

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Script started on {}'.format(time.ctime()))

    main()

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Script finished on {}'.format(time.ctime()))
