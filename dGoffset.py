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
path = './absolute_dGoffset/'
SDF_dr = './datasets/sdffiles/'

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


def main():

    mordred_df = get_descriptors()


def get_descriptors():

    save_loc = path + 'features_X/mordred_descriptor_output/mordred_descriptors.csv'

    if os.path.exists(save_loc):
        mordred_df = pd.read_csv(save_loc)
    else:
        descriptors_raw = open(absolute_dGoffset_path + 'features_X/mordred_descriptors/all_descriptors.txt', 'r')
        descriptors_raw_list = [line.split('\n') for line in descriptors_raw.readlines()]
        descriptors_list = [desc[0] for desc in descriptors_raw_list]
        print('Number of descriptors:', str(len(descriptors_list)))

        # set up feature calculator
        print("Generating features...")
        calc = Calculator(descriptors, ignore_3D=False)

        # Supply SDF
        suppl = [sdf for sdf in glob.glob(SDF_dr + '*.sdf')]

        # Empty DataFrame containing only descriptor names as headings
        mordred_df = pd.DataFrame(columns=descriptors_list)

        # generate features
        ID_lst = []
        for mol in suppl:
            ID = mol.strip(SDF_dr)
            ID_lst.append(ID)
            feat = calc.pandas(Chem.SDMolSupplier(mol))
            mordred_df = mordred_df.append(feat, ignore_index=True, sort=False)

        # Insert IDs as new column with column index = 0
        mordred_df.insert(0, 'ID', ID_lst)

    return mordred_df


if __name__ == '__main__':
    main()
    print('############### Finish ###############')
