# General:
import pandas as pd
import os

# Path variables
path = './'
datasets_dr = '../'


def main():

    # label generation
    get_labels()


def get_labels():

    # Load in FreeSolve database.
    freesolv_df = pd.read_csv(datasets_dr + 'freesolv_database.txt', sep='; ', engine='python')

    # Column names
    freesolv_id = freesolv_df.loc[:, 'compound id (and file prefix)']
    exp_val = freesolv_df.loc[:, 'experimental value (kcal/mol)']
    exp_err = freesolv_df.loc[:, 'experimental uncertainty (kcal/mol)']
    calc_val = freesolv_df.loc[:, 'Mobley group calculated value (GAFF) (kcal/mol)']
    calc_err = freesolv_df.loc[:, 'calculated uncertainty (kcal/mol)']

    # New nested list containing IDs and offsets.
    offsets = []
    for name, exp, err1, calc, err2 in zip(freesolv_id, exp_val, exp_err, calc_val, calc_err):
        offset = exp - calc
        error = (err1 ** 2 + err2 ** 2) ** 0.5
        offsets.append([name, offset, round(error, 3)])

    # Experimental offsets with uncertainties.
    exp_offset_with_errors_df = pd.DataFrame(offsets, columns=['ID', 'dGoffset (kcal/mol)', 'uncertainty (kcal/mol)'])

    # Experimental offsets only.
    exp_offset = exp_offset_with_errors_df.drop(columns=['uncertainty (kcal/mol)'])
    exp_offset = exp_offset.set_index('ID')

    # Save to CSV.
    save_loc = path + 'experimental_labels.csv'
    save_csv(exp_offset, save_loc)

    return exp_offset


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
