# General:
import pandas as pd
import numpy as np
import os
import glob

# SVM:
from sklearn.decomposition import PCA
from sklearn import preprocessing

# RDKit:
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdMolDescriptors

# Mordred descriptors:
from mordred import Calculator, descriptors

# Path variables:
datasets_dr = '../'
sdf_dr = datasets_dr + 'sdffiles/'

# PCA parameter:
pca_threshold = 0.95  # Keeps n dimensions for x variance explained


def main():

    # feature generation
    mordred_df = get_descriptors()
    fp_df = get_fingerprints()
    compiled_X_df = mordred_df.join(fp_df, on='ID')
    numeric_X = check_dataframe_is_numeric(compiled_X_df)
    float_X = convert_to_float(numeric_X)
    normalised_X = normalise_datasets(float_X)
    reduce_features(normalised_X, pca_threshold)


def reduce_features(normalised_collection, pca_threshold):
    """Returns PCA reduced DataFrame according to a pca_threshold parameter.
    Original columns with the highest contribution to PCX are written to CSV."""

    print('Computing PCA, reducing features up to ' + str(round(pca_threshold * 100, 5)) + '% VE...')
    training_data = normalised_collection

    # Initialise PCA object, keep components up to x% variance explained:
    PCA.__init__
    pca = PCA(n_components=pca_threshold)

    # Fit to and transform training set.
    train_post_pca = pd.DataFrame(pca.fit_transform(training_data))

    # Reset column names to PCX.
    pca_col = np.arange(1, len(train_post_pca.columns) + 1).tolist()
    pca_col = ['PC' + str(item) for item in pca_col]
    train_post_pca.columns = pca_col
    train_post_pca.index = training_data.index

    print('Number of PCA features after reduction: ' + str(len(train_post_pca.columns)))

    def recovery_pc(normalised_collection, pca_threshold):

        print('Computing PCA, reducing features up to ' + str(round(pca_threshold * 100, 5)) + '% VE...')
        training_data = normalised_collection

        # Normalise data.
        data_scaled = pd.DataFrame(preprocessing.scale(training_data), columns=training_data.columns)

        # Initialise PCA object, keep components up to x% variance explained:
        PCA.__init__
        pca = PCA(n_components=pca_threshold)
        pca.fit_transform(data_scaled)

        index = list(range(1, len(train_post_pca.columns) + 1))
        index = ['PC{}'.format(x) for x in index]

        return_df = pd.DataFrame(pca.components_, columns=data_scaled.columns, index=index)

        return return_df

    # Adapted from https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in
    # -pca-with-sklearn
    recovered_pc = recovery_pc(normalised_collection, pca_threshold)

    # List of column names with highest value in each row.
    recovered_pc_max = recovered_pc.idxmax(axis=1)

    # Recovery 'PCX' indexing.
    pc_index = recovered_pc_max.index.tolist()

    # Write feature names to list.
    pc_feature = recovered_pc_max.values.tolist()

    # Write to DataFrame.
    recovered_pc_dict = {'PCX': pc_index, 'Highest contributing feature': pc_feature}
    recovered_pc_df = pd.DataFrame(recovered_pc_dict)

    # Save recovered PCs to CSV
    save_loc = 'recovered_PCs.csv'
    save_csv(recovered_pc_df, save_loc)

    # Save reduced features to CSV
    save_loc = 'reduced_features.csv'
    save_csv(train_post_pca, save_loc)

    return train_post_pca


def normalise_datasets(dataframe):
    """Returns a normalised DataFrame"""

    # Calculate statistics, compute Z-scores, clean.
    print('Normalising dataframe...')
    stat = dataframe.describe()
    stat = stat.transpose()

    def norm(x):
        return (x - stat['mean']) / stat['std']

    # Normalise and return separately.
    normed_data = norm(dataframe).fillna(0).replace([np.inf, -np.inf], 0.0)

    print('Completed normalising dataframe.')
    return normed_data


def convert_to_float(dataframe):
    """Returns a DataFrame with all cells are converted to floats"""

    print('Converting dataframe to float...')
    float_df = dataframe.apply(pd.to_numeric).astype(float).sample(frac=1)
    float_df = float_df.rename(columns={'dGhydr (kcal/mol)': 'dGoffset (kcal/mol)'})

    print('Completed converting dataframe to flaot.')
    return float_df


def check_dataframe_is_numeric(dataframe):
    """Returns new DataFrame with non-numeric columns removed.
    Dropped columns are saved to CSV."""

    columns_dropped = []

    print('Checking dataframe is numeric...')
    for col in dataframe.columns:
        for index, x in zip(dataframe.index, dataframe.loc[:, col].tolist()):
            try:
                float(x)
            except ValueError:
                columns_dropped.append([col, index, x])
                dataframe = dataframe.drop(columns=col)
                break

    # Save dropped columns to CSV.
    dropped_col_df = pd.DataFrame(columns_dropped, columns=['column dropped', 'at ID', 'non-numeric value'])
    save_loc = 'dropped_features.csv'
    save_csv(dropped_col_df, save_loc)

    print('Number of columns dropped:', (len(columns_dropped)))
    return dataframe


def get_fingerprints():
    """Returns DataFrame and saves CSV of all calculated RDKit fingerprints for all
    SDF files in the SDF directory (SDF_dr) specified in the path variables."""

    fp_table = []
    for sdf in glob.glob(sdf_dr + '*.sdf'):

        fp_row = []

        # Append ligand ID.
        fp_row.append(sdf.strip(sdf_dr).strip('*.sdf'))

        # Setup fingerprint.
        mol = Chem.rdmolfiles.SDMolSupplier(sdf)[0]
        mol.UpdatePropertyCache(strict=False)

        # Calculate fingerprint.
        fp = rdMolDescriptors.GetHashedAtomPairFingerprint(mol, 256)
        for x in list(fp): fp_row.append(x)

        fp_table.append(fp_row)

    # Column names:
    id_col = ['ID']
    fp_col = np.arange(0, 256).tolist()
    fp_col = [id_col.append("pfp" + str(item)) for item in fp_col]

    fp_df = pd.DataFrame(fp_table, columns=id_col)
    fp_df = fp_df.set_index('ID')

    # Save to CSV.
    save_loc = 'calculated_fingerprints.csv'
    save_csv(fp_df, save_loc)

    print('Completed calculating fingerprints.')
    return fp_df


def get_descriptors():
    """Returns DataFrame and saves CSV of all calculated Mordred descriptors for all
    SDF files in the SDF directory (SDF_dr) specified in the path variables."""

    save_loc = 'calculated_mordred_descriptors.csv'
    if os.path.exists(save_loc):
        mordred_df = pd.read_csv(save_loc, index_col='ID')
        print('Calculated Mordred descriptors loaded in.')
    else:
        # Read in mordred descriptors to be calculated. In this case, all descriptors.
        descriptors_raw = open(datasets_dr + 'all_mordred_descriptors.txt', 'r')
        descriptors_raw_list = [line.split('\n') for line in descriptors_raw.readlines()]
        descriptors_list = [desc[0] for desc in descriptors_raw_list]
        print('Number of descriptors:', str(len(descriptors_list)))

        # Setup feature calculator.
        print('Calculating Mordred descriptors...')
        calc = Calculator(descriptors, ignore_3D=False)

        # Supply SDF.
        suppl = [sdf for sdf in glob.glob(sdf_dr + '*.sdf')]

        # Empty DataFrame containing only descriptor names as headings.
        mordred_df = pd.DataFrame(columns=descriptors_list)

        # Generate features.
        for mol in suppl:
            feat = calc.pandas(Chem.SDMolSupplier(mol))
            mordred_df = mordred_df.append(feat, ignore_index=True, sort=False)

        # Insert IDs as index.
        ID_lst = [mol.strip(sdf_dr) for mol in suppl]
        mordred_df.set_index(ID_lst)

        # Save to CSV.
        save_csv(mordred_df, save_loc)

    return mordred_df


def save_csv(dataframe, pathname):

    if os.path.exists(pathname):
        os.remove(pathname)
        dataframe.to_csv(path_or_buf=pathname, index=True)
        print('Existing file overwritten.')
    else:
        dataframe.to_csv(path_or_buf=pathname, index=True)
    print('Completed writing {}.'.format(pathname))


if __name__ == '__main__':
    main()
