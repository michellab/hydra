# General imports
import pandas as pd
import numpy as np
import os
import csv
import subprocess
import time
import shutil

# SciKit-Optimize:
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

#
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import pickle

# Misc. imports:
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from scipy import stats
import statistics
import pickle


def main():

    # global variables
    dataset_path = '~/Dropbox/FreeSolv/dGlearn-FreeSolv-master/datasets/train_compiled/dGhydr_train.csv'
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

    split = 'dG(hydr)'
    translated_subject = 'absolute'

    # construct raw dataset
    raw_data = pd.read_csv(dataset_path, index_col='ID')
    # remove columns with string values
    numeric_data, columns_dropped = check_dataframe_is_numeric(raw_data)
    # convert all values to float
    float_data = numeric_data.apply(pd.to_numeric).astype(float).sample(frac=1)
    float_data = float_data.rename(columns={'dGhydr (kcal/mol)': 'dGoffset (kcal/mol)'})
    # normalise data and separate labels before PCA
    normalised_X, y_tmp = normalise_and_split_datasets(float_data)
    # perform PCA on features alone
    reduced_X = reduce_features(normalised_X, PCA_threshold)
    # recombine reduced features with labels with the correct indexing
    dataset = pd.concat([reduced_X, y_tmp], axis=1)
    # perform 5-fold cross-validation
    kfolds = split_dataset(dataset, n_splits, random_state)

    # Kfold data structure
    for fold in kfolds:
        dataframe = fold
        train_postPCA_df, test_postPCA_df, train_labels_df, test_labels_df = dataframe[0][0], dataframe[0][1], \
                                                                             dataframe[1][0], dataframe[1][1]
        train_postPCA = train_postPCA_df.astype(np.float32).values
        test_postPCA = test_postPCA_df.astype(np.float32).values
        train_labels = train_labels_df.astype(np.float32).values
        test_labels = test_labels_df.astype(np.float32).values

    # start log file
    # initiate empty DF to fill with cumulative minima
    cumulative_MAEs = pd.DataFrame()
    cumulative_MAEtauR_CV = pd.DataFrame()

    # clean slate opt_output:
    if os.path.exists("./opt_output"):
        subprocess.call('rm ./opt_output/*', shell=True)
    if not os.path.exists("./opt_output"):
        os.mkdir("./opt_output")

    # initiate log file:
    with open("opt_output/logfile.txt", "w") as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["###########Starting tb-CV BO.###########"])
        writer.writerow(["PCA threshold: " + str(PCA_threshold)])
        writer.writerow(["n replicates: " + str(replicates)])
        writer.writerow(["n models in ensemble: " + str(ensemble_size)])
        writer.writerow(["n calls (BO): " + str(n_calls)])
        writer.writerow(["Started program at: " + time.ctime()])

    bucket_df = pd.DataFrame()
    mae_results_per_fold = [["Subject", "MAE", "Replicate"]]
    MAEtauR_results_per_fold = [["Subject", "Correlation Coefficient", "Dataset", "Correlation metric"]]

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(time.ctime())

    # train SVR
    for fold in kfolds:
        MAEs_per_fold = []
        models_per_replicate = []
        for i in range(replicates):
            # run tb-CV:
            # reset MAEMAD startpoint per replicate:
            startpoint_MAEMAD = startpoint_BO
            OptimizeResult, top_performer = svr(fold, 'dGhydr', i)

            models_per_replicate.append(top_performer)

            # construct, cummin and concatenate results of this replicate to the other replicates in the loop:
            split_columns = {
                "Dataset": str(split),
                "MAE/MAD": OptimizeResult.func_vals,
                "Subject": translated_subject}
            result_df = pd.DataFrame(split_columns).cummin()
            bucket_df = pd.concat([bucket_df, result_df])
            # tag data with the dataset type (i.e. descriptor set), add to complete results:
            bucket_df["Dataset"] = str(split)
            cumulative_MAEs = pd.concat([cumulative_MAEs, bucket_df])

            # retrieve statistics for this replicate:
            tau = top_performer[3]
            r_value = top_performer[4]
            MAE = top_performer[0]

            MAEtauR_results_per_fold.append([translated_subject, r_value, split, "Pearson's-r"])
            MAEtauR_results_per_fold.append([translated_subject, tau, split, "Kendall's-tau"])
            MAEtauR_results_per_fold.append([translated_subject, MAE, split, "MAE/MAD"])

            # write update to log file:
            with open("opt_output/logfile.txt", "a") as file:
                writer = csv.writer(file, delimiter='\t')
                writer.writerow(["Finished " + translated_subject + ", dataset " + split + ", replicate " + str(
                    i + 1) + " at " + str(time.ctime())])

    # make ensemble of best models; pick n replicates' top performing models:
    models_per_replicate = sorted(models_per_replicate, key=lambda x: x[0])

    ensemble_collection = models_per_replicate[:ensemble_size]

    i = 1
    for best_model_collection in ensemble_collection:

        opt_replicate = str(best_model_collection[2])
        result_internal = MAE

        nested_list_result_external = best_model_collection[5]

        # For this best model, retrieve model files, plot internal validation and write external validation to file:
        if not os.path.exists("./opt_output"):
            os.mkdir("./opt_output")

        # with the known optimal replicate #, isolate model files from opt_tmp and move to opt_output:
        # rename so that name contains name of the feature set instead of the replicate:
        os.rename(
            "opt_tmp/" + opt_replicate + "_ALFRESCO_TopPerform_SVM.svm",
            "opt_output/model" + str(i) + "_" + split + "_" + translated_subject + "_ALFRESCO_TopPerform_SVM.svm"
        )

        i += 1
    # to keep things clean, remove ./opt_tmp:
    shutil.rmtree("./opt_tmp/")

    # write internal validation MAEMAD value:
    internal_val = pd.DataFrame([result_internal], columns=["val_loss"])

    internal_val.to_csv("opt_output/" + str(split) + "_" + str(translated_subject) + "_TopPerformer_internalVal_df.csv")

    # write external validation DF:
    with open("opt_output/" + str(split) + "_" + str(translated_subject) + "_TopPerformer_externalVal_df.csv",
              "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Perturbation", "Experimental ddGoffset (kcal/mol)", "Predicted ddGoffset (kcal/mol)", "Subject"])
        for row in nested_list_result_external:
            writer.writerow(row + [translated_subject])

    MAEs_CV = pd.DataFrame(mae_results_per_fold[1:], columns=mae_results_per_fold[0])
    MAEtauR_CV = pd.DataFrame(MAEtauR_results_per_fold[1:], columns=MAEtauR_results_per_fold[0])
    cumulative_MAEtauR_CV = pd.concat([cumulative_MAEtauR_CV, MAEtauR_CV])

    cumulative_MAEtauR_CV.to_csv("opt_output/tb-CV_MAEtauR_outputs.csv", index=False)
    cumulative_MAEs.to_csv("opt_output/tbCV_BO_MAE.csv")
    print("Success, wrote all files to opt_output/.")


def svr(dataframe, dataset_name, iteration):
    model_bucket = []

    #     [[train_set, test_set], [train_labels, test_labels]]

    # Retrieve datasets, convert to float32 for RF:
    train_postPCA_df, test_postPCA_df, train_labels_df, test_labels_df = dataframe[0][0], dataframe[0][1], dataframe[1][
        0], dataframe[1][1]
    train_postPCA = train_postPCA_df.astype(np.float32).values
    test_postPCA = test_postPCA_df.astype(np.float32).values
    train_labels = train_labels_df.astype(np.float32).values
    test_labels = test_labels_df.astype(np.float32).values

    # Set hyperparameter ranges, append to list:
    dim_param_C = Categorical(categories=list(np.logspace(-3, 2, 6, dtype="float32")), name="param_C")
    dim_param_gamma = Categorical(categories=list(np.logspace(-3, 2, 6, dtype="float32")), name="param_gamma")
    dim_param_epsilon = Categorical(categories=list(np.logspace(-3, 2, 6, dtype="float32")), name="param_epsilon")

    dimensions = [dim_param_C, dim_param_gamma, dim_param_epsilon]

    @use_named_args(dimensions=dimensions)
    def fitness(param_C, param_gamma, param_epsilon):
        # Create the svm with these hyper-parameters:

        svm_estimator = SVR(gamma=param_gamma, C=param_C, epsilon=param_epsilon)
        svm_estimator.fit(train_postPCA, train_labels)

        prediction_list = svm_estimator.predict(test_postPCA)

        # calculate some statistics on test set:
        MAE = mean_absolute_error(test_labels, prediction_list)
        MAD_testset = test_labels_df.mad()

        MAEMAD = MAE / MAD_testset
        print("MAE/MAD:", MAEMAD)

        perts_list = test_labels_df.index.tolist()
        exp_list = test_labels_df.values.tolist()

        slope, intercept, r_value, p_value, std_err = stats.linregress(prediction_list, exp_list)
        tau, p_value = stats.kendalltau(prediction_list, exp_list)

        # For plotting test set correlations:
        tuples_result = list(zip(perts_list, exp_list, prediction_list))
        nested_list_result = [list(elem) for elem in tuples_result]

        # Append data with best performing model.
        # Data contains the MAE/MAD score, protein target, iteration,
        # tau, r value, the keras DNN model, the internal validation plot
        # and the data for external validation:

        global startpoint_MAEMAD

        if MAEMAD < startpoint_MAEMAD:
            startpoint_MAEMAD = MAEMAD
            model_bucket.append([MAEMAD, dataset_name, iteration, tau, r_value, nested_list_result])

            # # write all model files:
            if not os.path.exists("./opt_tmp"):
                os.makedirs("./opt_tmp")

            with open("./opt_tmp/" + str(iteration) + "_ALFRESCO_TopPerform_SVM.svm", "wb") as file:
                pickle.dump(svm_estimator, file)

        return MAEMAD

    # Bayesian Optimisation to search through hyperparameter space.
    # Prior parameters were found by manual search and preliminary optimisation loops.
    # For running just dataset 13x500 calls, optimal hyperparameters from 150 calls were used as prior.
    default_parameters = [1.0, 1.0, 1.0]
    print("###########################################")
    print("Created model, optimising hyperparameters..")

    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI',  # Expected Improvement.
                                n_calls=n_calls,
                                x0=default_parameters)

    print("###########################################")
    print("Concluded optimal hyperparameters:")
    print(search_result.x)

    print("###########################################")

    # return skopt object and highest scoring model for this replicate:
    return search_result, model_bucket[-1]


def split_dataset(dataset, n_splits, random_state):
    """KFold implementation for pandas DataFrame.
    (https://stackoverflow.com/questions/45115964/separate-pandas-dataframe-using-sklearns-kfold)"""

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    kfolds = []
    global offset_col_name

    for train, test in kf.split(dataset):
        training = dataset.iloc[train]
        train_labels = training[offset_col_name]
        train_set = training.drop(offset_col_name, axis=1)

        testing = dataset.iloc[test]
        test_labels = testing[offset_col_name]
        test_set = testing.drop(offset_col_name, axis=1)

        kfolds.append(
            [[train_set, test_set],
             [train_labels, test_labels]]
        )

    return kfolds


def reduce_features(normalised_collection, pca_threshold):

    print('Computing PCA, reducing features up to ' + str(round(pca_threshold * 100, 5)) + '% VE..')
    training_data = normalised_collection

    # Initialise PCA object, keep components up to x% variance explained:
    PCA.__init__
    pca = PCA(n_components=pca_threshold)

    # Fit to and transform training set:
    train_post_pca = pd.DataFrame(pca.fit_transform(training_data))

    print('# of PCA features after reduction: ' + str(len(train_post_pca.columns)))

    train_post_pca.index = training_data.index
    # pickle pca object to file so that external test sets can be transformed accordingly
    # (see https://stackoverflow.com/questions/42494084/saving-large-data-set-pca-on-disk
    # -for-later-use-with-limited-disc-space)
    # pickle.dump(pca, open('./opt_output/pca_featureset', 'wb'))

    return train_post_pca  # return list with test_post_pca when needed


def normalise_and_split_datasets(collection):

    # process input dataset
    train_dataset = collection

    print('Normalising...')
    # Calculate statistics, compute Z-scores, clean:
    stats = train_dataset.describe()

    stats.pop('dGoffset (kcal/mol)')
    stats.pop('uncertainty (kcal/mol)')
    stats = stats.transpose()

    train_labels = train_dataset.pop('dGoffset (kcal/mol)')
    train_dataset.pop('uncertainty (kcal/mol)')

    def norm(x):
        return (x - stats['mean']) / stats['std']

    # Normalise and return separately:
    normed_train_data = norm(train_dataset).fillna(0).replace([np.inf, -np.inf], 0.0)

    return [normed_train_data, train_labels]


def check_dataframe_is_numeric(dataframe):
    """Iterate over all columns and check if numeric.

    Returns:
    New DataFrame with removed"""

    columns_dropped = 0
    columns_dropped_lst = []

    for col in dataframe.columns:
        for x in dataframe.loc[:, col]:
            try:
                float(x)
            except ValueError:
                columns_dropped_lst.append(col)
                columns_dropped += 1
                dataframe = dataframe.drop(columns=col)
                break

    print('Number of columns dropped:', (columns_dropped))
    return dataframe, columns_dropped_lst


if __name__ == '__main__':
    main()
