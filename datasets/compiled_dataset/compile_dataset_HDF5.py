import pandas as pd
import os
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

    # Load in features and labels and sort by index
    reduced_X = pd.read_csv(datasets_dr + 'features_X/reduced_features.csv', index_col='ID').sort_index()
    true_y = pd.read_csv(datasets_dr + 'labels_y/experimental_labels.csv', index_col='ID').sort_index()
    full_dataset = pd.concat([reduced_X, true_y], axis=1, sort=False).sort_index()

    # split into training and external testing sets
    train_X, test_X = split_train_test(reduced_X)
    train_y, test_y = split_train_test(true_y)
    train_full, test_full = split_train_test(full_dataset)

    # absolute data sets
    create_absolute_train_test(train_X, train_y, 'train')
    create_absolute_train_test(test_X, test_y, 'test')

    # # relative data sets
    create_relative_train_test(train_full, 'train')
    create_relative_train_test(test_full, 'test')


def create_relative_train_test(dataframe, set_type):
    """col - row, where col and row notation is taken from matrices nomenclature.
    Index values must be Mobley IDs.
    set_type: 'train' or 'test'
    matrix diagonal is not omitted
    note: label column will still be called 'dGoffset (kcal/mol)' and not 'ddGoffset (kcal/mol)'"""

    # setup HDF5 file
    hdf = pd.HDFStore(datasets_dr + 'd' + set_type + '_data.h5')
    hdf.put('relative', pd.DataFrame(), format='table', data_columns=True)

    for id1, col in tqdm(dataframe.iterrows(),
                         total=dataframe.shape[0],
                         desc='Writing d{}_data.h5'.format(set_type),
                         unit_scale=True, leave=True, ncols=100):

        for id2, row in dataframe.iterrows():
            df = pd.DataFrame([col - row], index=[str(id1) + '~' + str(id2)])
            if not df.iloc[0, 0] == 0:
                hdf.append(key='relative', value=df, format='table', index=False)

    hdf.close()


def create_absolute_train_test(features, labels, set_type):
    """set_type: 'train' or 'test'"""

    # setup HDF5 file
    hdf = pd.HDFStore(datasets_dr + set_type + '_data.h5')
    hdf.put('absolute', pd.DataFrame(), format='table', data_columns=True)

    for (id1, X), (id2, y) in tqdm(zip(features.iterrows(), labels.iterrows()),
                         total=features.shape[0],
                         desc='Writing {}_data.h5'.format(set_type),
                         unit_scale=True, leave=True, ncols=100):

        # write row to HDF5 file
        row = pd.concat([pd.DataFrame([X]), pd.DataFrame([y])], axis=1)
        hdf.append(key='absolute', value=row, format='table', index=False)

        # copy SDF files
        sdf = str(id1) + '.sdf'
        if set_type == 'train':
            shutil.copyfile(SDF_dr + sdf, train_dr + sdf)
        elif set_type == 'test':
            shutil.copyfile(SDF_dr + sdf, test_dr + sdf)

    hdf.close()


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
