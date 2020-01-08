# General:
import pandas as pd
import os
import shutil
import glob

# Path variables:
path = './'
datasets_dr = '../'
SDF_dr = datasets_dr + 'sdffiles/'
freesolv_loc = datasets_dr + 'freesolv_database.txt'
train_dr = datasets_dr + 'train_dr/'
test_dr = datasets_dr + 'test_dr/'

# Gloval variables:
offset_col_name = 'dGoffset (kcal/mol)'
# Load in FreeSolve.
freesolv_df = pd.read_csv(freesolv_loc, sep='; ', engine='python')
# SAMPl4_Guthrie experimental reference in FreeSolv.
SAMPL4_Guthrie_ref = 'SAMPL4_Guthrie'
# Experimental reference column name.
exp_ref_col = 'experimental reference (original or paper this value was taken from)'


def main():

    # Load in features and labels.
    reduced_X = pd.read_csv(datasets_dr + 'features_X/reduced_features.csv', index_col='ID')
    true_y = pd.read_csv(datasets_dr + 'labels_y/experimental_labels.csv', index_col='ID')

    # Datasets for absolute predictions.
    full_dataset = get_full_dataset(reduced_X, true_y)
    train_df, test_df = separate_train_test(full_dataset)

    # Datasets for relative predictions.

    # # uncomment below for testing purposes
    # train_df = train_df.head(100)

    dtrain_df = calc_relative_dataframe(train_df)
    save_csv(dtrain_df, datasets_dr + 'dtrain_data.csv')

    dtest_df = calc_relative_dataframe(test_df)
    save_csv(dtest_df, datasets_dr + 'dtest_data.csv')


def calc_relative_dataframe(dataframe):
    """col - row, where col and row notation is taken from matrices nomenclature.
    Index values must be Mobley IDs."""

    # save original column names
    col_names = dataframe.columns.values.tolist()

    # compute relative IDs
    ID = dataframe.index.tolist()
    dID = [str(col) + '~' + str(row) for col in ID for row in ID]

    # compute relative labels
    y = dataframe.pop('dGoffset (kcal/mol)').tolist()
    dy = [col - row for col in y for row in y]

    # compute relative features
    X = dataframe.values.tolist()
    dX = [[col[i] - row[i] for i in range(len(dataframe.columns))]
          for col in X for row in X]

    # construct features dataframe using the dictionary method
    ddict = {ID: feat for ID, feat in zip(dID, dX)}
    ddataset = pd.DataFrame.from_dict(ddict, orient='index')

    # add labels
    ddataset['ddGoffset (kcal/mol)'] = dy
    ddataset.columns = col_names

    #  drop row if any values in the row equal zero
    ddataset = ddataset.loc[~(ddataset == 0).all(axis=1)]

    return ddataset


def separate_train_test(full_dataset):

    # List comprehension for all non-SAMPL4_Guthrie entries.
    train_ids = [freesolv_df.iloc[i][0]
                 for i in range(len(freesolv_df))
                 if freesolv_df.loc[i, exp_ref_col] != SAMPL4_Guthrie_ref]

    # List comprehension for all SAMPL4_Guthrie entries.
    test_ids = [freesolv_df.iloc[i][0]
                for i in range(len(freesolv_df))
                if freesolv_df.loc[i, exp_ref_col] == SAMPL4_Guthrie_ref]

    print('Creating training set...')
    print('\tfull_dataset\n')
    print(full_dataset)
    train_df = full_dataset.drop(test_ids)
    save_csv(train_df, datasets_dr + 'train_data.csv')
    save_sdf(train_ids, train_dr)

    print('Creating testing set...')
    test_df = full_dataset.drop(train_ids)
    save_csv(test_df, datasets_dr + 'test_data.csv')
    save_sdf(test_ids, test_dr)

    return train_df, test_df


def get_full_dataset(feature_df, label_df):

    print('Generating full dataset...')
    full_dataset = pd.concat([feature_df, label_df], axis=1, sort=False)

    print('feature_df')
    print(feature_df)
    print('label_df')
    print(label_df)

    # Save to CSV.
    save_loc = path + 'full_dataset.csv'
    save_csv(full_dataset, save_loc)

    return full_dataset


def save_sdf(ID_lst, dr_path):

    # Create directory
    if os.path.isdir(dr_path):
        shutil.rmtree(dr_path)
        print('Existing directory overwritten.')
        os.mkdir(dr_path)
    else:
        os.mkdir(dr_path)

    for entry in ID_lst:
        sdf = entry + '.sdf'
        shutil.copyfile(SDF_dr + sdf, dr_path + sdf)

    # Check the number of ligands found is correct.
    print('Number of entires in {}: {}'.format(dr_path, len(glob.glob(dr_path + '*.sdf'))))


def save_csv(dataframe, pathname):

    if os.path.exists(pathname):
        os.remove(pathname)
        dataframe.to_csv(path_or_buf=pathname, index=True)
        print('Existing file overwritten.')
    else:
        dataframe.to_csv(path_or_buf=pathname, index=True)
    print('Completed writing {}.csv.'.format(pathname))


if __name__ == '__main__':
    main()
