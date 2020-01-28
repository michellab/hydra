# General:
import pandas as pd
import os
import numpy as np

# Path variables
path = './'
datasets_dr = '../'

# Load in FreeSolve database.
freesolv_df = pd.read_csv(datasets_dr + 'freesolv_database.txt', sep='; ', engine='python')

def main():

    # label generation
    get_labels()


def get_labels():

    # Mobley IDs
    freesolv_id = freesolv_df.loc[:, 'compound id (and file prefix)']
    # Experimentally determined dGhydr
    exp_val = freesolv_df.loc[:, 'experimental value (kcal/mol)']
    # Associated error
    exp_err = freesolv_df.loc[:, 'experimental uncertainty (kcal/mol)']
    # Computationally determined dGhydr
    calc_val = freesolv_df.loc[:, 'Mobley group calculated value (GAFF) (kcal/mol)']
    # Associated error
    calc_err = freesolv_df.loc[:, 'calculated uncertainty (kcal/mol)']
    # null computationally determined dGhydr
    calc_null = np.zeros(len(calc_val))

    # nested dict containing ID keys with offset and error values
    offset = {name: [exp - calc, (err1 ** 2 + err2 ** 2) ** 0.5]
                      for name, exp, err1, calc, err2 in zip(freesolv_id, exp_val, exp_err, calc_val, calc_err)}

    # nested dict containing ID keys with null offsets and error values
    null = {name: [exp - null, (err1 ** 2 + null ** 2) ** 0.5]
            for name, exp, err1, null, err2 in zip(freesolv_id, exp_val, exp_err, calc_null, calc_null)}

    offset_df = pd.DataFrame.from_dict(data=offset, orient='index',
                                        columns=['dGoffset (kcal/mol)', 'uncertainty (kcal/mol)'])
    offset_df.index.name = 'ID'

    null_df = pd.DataFrame.from_dict(data=null, orient='index',
                                     columns=['null dGoffset (kcal/mol)', 'null uncertainty (kcal/mol)'])
    null_df.index.name = 'ID'

    # Save to CSV.
    save_csv(offset_df, path + 'experimental_labels.csv')
    save_csv(null_df, path + 'null_experimental_labels.csv')

    return offset_df, null_df


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
