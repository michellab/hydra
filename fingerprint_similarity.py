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
absolute_dGoffset_path = './absolute_dGoffset/'

# dataset_path = '~/Dropbox/FreeSolv/dGlearn-FreeSolv-master/datasets/train_compiled/dGhydr_train.csv'
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

train_df = pd.read_csv(absolute_dGoffset_path + 'train_data.csv', index_col='Unnamed: 0')
train_dr = absolute_dGoffset_path + 'train_dr/'

test_dr = absolute_dGoffset_path + 'test_dr/'
test_df = pd.read_csv(absolute_dGoffset_path + 'test_data.csv', index_col='Unnamed: 0')
test_ID = test_df.index


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


kfolds = split_dataset(train_df, n_splits, random_state)


# retrieve IDs
train_IDs = kfolds[0][0][0].index.tolist()
validate_IDs = kfolds[0][0][1].index.tolist()
test_IDs = test_ID.tolist()

# retrieve SDFs
train_suppl = [Chem.SDMolSupplier(train_dr + str(sdf) + '.sdf')
               for sdf in train_IDs]

valdtn_suppl = [Chem.SDMolSupplier(train_dr + str(sdf)  + '.sdf')
                for sdf in validate_IDs]

test_suppl = [Chem.SDMolSupplier(test_dr + str(sdf)  + '.sdf')
              for sdf in test_IDs]


# generate fingerprints
train_fp = [Chem.RDKFingerprint(mol) for mol in train_suppl]
valdtn_fp = [Chem.RDKFingerprint(mol) for mol in valdtn_suppl]
test_fp = [Chem.RDKFingerprint(mol) for mol in test_suppl]


# compare fingerprints
test_train_similarity = [DataStructs.FingerprintSimilarity(test_mol, train_mol)
                         for test_mol in test_fp
                         for train_mol in train_fp]

test_valdtn_similarity = [DataStructs.FingerprintSimilarity(test_mol, valdtn_mol)
                          for test_mol in test_fp
                          for valdtn_mol in valdtn_fp]

print(test_train_similarity)
