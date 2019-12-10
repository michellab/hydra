#!/usr/bin/env python
# -*- coding: utf-8 -*-

###################################
##### ALFRESCO RF builder #####
###################################



# TF-related imports & some settings to reduce TF verbosity:
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""	# current workstation contains 4 GPUs; exclude 1st
import tensorflow as tf 
from tensorflow import keras
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.ops import resources

# SciKit-Optimize:
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from tensorflow.python.keras import backend as K
from skopt.utils import use_named_args

# General imports:
import glob
import shutil
import subprocess
import numpy as np
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import matplotlib.cbook
import time
start_time = "Start: "+str(time.ctime())
import warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
np.seterr(divide='ignore', invalid='ignore')

# Misc. imports:
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from scipy import stats
import statistics
import pickle

print(
"\n"
"\n"
"##############################################################\n"
"######################  ALFRESCO - Build #####################\n"
"##############################################################\n"
"############ Target-based-Cross-Validation SVM ###############\n"
"# model generator using Bayesian hyperparameter optimisation #\n"
"##############################################################\n")
print("Imports complete.\n")

# set dataset to optimise on:
dataset_paths = {
  '../datasets/trainingsets_compiled/dataset_1.csv': "1",
 '../datasets/trainingsets_compiled/dataset_2.csv': "2",
 '../datasets/trainingsets_compiled/dataset_3.csv': "3",
 '../datasets/trainingsets_compiled/dataset_12.csv': "12",
 '../datasets/trainingsets_compiled/dataset_13.csv': "13",
 '../datasets/trainingsets_compiled/dataset_23.csv': "23",
 '../datasets/trainingsets_compiled/dataset_123.csv': "123",
  '../datasets/trainingsets_compiled/dataset_noise.csv': "Noise"
}

# list of ligand names in compiled dataset files:
# update when new targets are added!!

targets = [
			"HSP90",
			"jnk1",
			"FXR", 
			"ACK1",
			"throm_jm", 
			"tyk", 
			"throm_schr",
			#"test", 
			"BACE", 		
			"cdk2",
			"mcl1",
			"ptp1b"
			]

# set data processing configurations:
PCA_threshold = 0.95				# Keeps n dimensions for x variance explained
replicates = 30				# Number of replicates per subject model
n_calls = 40						# Number of Bayesian optimisation loops for hyperparameter optimisation, 40 is best for convergence, > 60 scales to very expensive
startpoint_BO = np.inf				# Point to consider top-performing model from (MAE/MAD); 1.0 = no improvement on test-set variance
ensemble_size = 10					# Amount of top-scoring models to retain per fold-dataset combination
runtime_estimation = replicates * 3.2 * len(dataset_paths)/60			

print(
	"Program is set with the following settings:\n"
	"Feature sets:", str(len(dataset_paths)),"\n"
	"Protein (i.e. perturbation) sets:", str(len(targets)),"\n"
	"PCA variance retained threshold = "+str(PCA_threshold)+"\n"			
	"Replicates = "+str(replicates)+"\n"			
	"Model ensemble size = "+str(ensemble_size)+"\n"
	"\n"
	)
if n_calls == 40:
	print("Estimated runtime:", str(round(runtime_estimation, 1)), "hours")
else:
	print("Could not estimate runtime because n_calls is not 40.")

def TranslateTargetNames(input_name):
	newname = input_name
	
	if "throm_jm" in newname:
		newname = "THROMBIN-JM"
	if "throm_schr" in newname:
		newname = "THROMBIN-SCHR"
	if "cdk2" in newname:
		newname = "CDK2"
	if "FXR" in newname:
		newname = "FXR"			
	if "ACK1" in newname:
		newname = "ACK1"
	if "tyk" in newname:
		newname = "TYK2"
	if "jnk1" in newname:
		newname = "JNK1"
	if "HSP90" in newname:
		newname = "HSP90"
	if "BACE" in newname:
		newname = "BACE"
	if "mcl1" in newname:
		newname = "MCL1"
	if "ptp1b" in newname:
		newname = "PTP1B"


	if "test" in newname:
		newname = "2ndBACE"
		
			
	return newname

def NormaliseDatasets(collection):
	# process input nested lists of datasets (per target):
	train_dataset = collection
	
	print("Normalising..")
	# Calculate statistics, compute Z-scores, clean:
	train_stats = train_dataset.describe()

	train_stats.pop("ddG_offset")
	train_stats = train_stats.transpose()

	train_labels = train_dataset.pop('ddG_offset')


	def norm(x):
		return (x - train_stats['mean']) / train_stats['std']

	# Normalise and return seperately:
	normed_train_data = norm(train_dataset).fillna(0).replace([np.inf, -np.inf], 0.0)
	
	
	return [normed_train_data, train_labels]



def ReduceFeatures(normalised_collection, PCA_threshold, split):
	print("Computing PCA, reducing features up to "+str(round(PCA_threshold*100, 5))+"% VE..")
	training_data = normalised_collection
	
	# Initialise PCA object, keep components up to x% variance explained:
	PCA.__init__
	pca = PCA(n_components=PCA_threshold)


	# Fit to and transform training set:			
	train_postPCA = pd.DataFrame(pca.fit_transform(training_data))

	print("# of PCA features after reduction: "+str(len(train_postPCA.columns)))

	train_postPCA.index = training_data.index
	# pickle pca object to file so that external test sets can be transformed accordingly (see https://stackoverflow.com/questions/42494084/saving-large-data-set-pca-on-disk-for-later-use-with-limited-disc-space)
	pickle.dump(pca, open("./opt_output/pca_featureset_"+str(split)+".p", "wb"))
	return train_postPCA	# return list with test_postPCA when needed


def SplitDatasets(dataset, targets, labels):
	print("Splitting data per target..")


	dataset = pd.concat([dataset, labels], axis=1)

	# split datasets by regex, store into dict:
	target_splits_dict = {}
	
	for target_name in targets:
		split_set = dataset[dataset.index.str.contains(target_name, regex=False)]
		target_splits_dict[target_name] = split_set

	# construct train-test combinations for all target datasets:
	CV_splits = []
	for key, test_set in target_splits_dict.items():
		training_set = [ rows for target, rows in target_splits_dict.items() if key != target ]
		training_set = pd.concat(training_set)


		train_labels = training_set["ddG_offset"]
		test_labels = test_set["ddG_offset"]
		
		training_set = training_set.drop("ddG_offset", axis=1)
		test_set = test_set.drop("ddG_offset", axis=1)

		CV_splits.append([[training_set, test_set], [train_labels, test_labels]])

	print("Done. Initialising tb-CV optimisation loops..")
	return CV_splits


def RF_TF(dataframe, dataset_name, iteration):

	model_bucket = []

	# Retrieve datasets, convert to float32 for RF:
	train_postPCA_df, test_postPCA_df, train_labels_df, test_labels_df = dataframe[0][0], dataframe[0][1], dataframe[1][0], dataframe[1][1]
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
    # Create the random forest with these hyper-parameters:

		
		svm_estimator = SVR(gamma=param_gamma, C=param_C, epsilon=param_epsilon)
		svm_estimator.fit(train_postPCA, train_labels)  
	
		prediction_list = svm_estimator.predict(test_postPCA)

		# calculate some statistics on test set:
		MAE = mean_absolute_error(test_labels, prediction_list)
		MAD_testset = test_labels_df.mad()

		MAEMAD = MAE/MAD_testset
		print("MAE/MAD:",MAEMAD)

		perts_list = test_labels_df.index.tolist()
		exp_list = test_labels_df.values.tolist()

		slope, intercept, r_value, p_value, std_err = stats.linregress(prediction_list, exp_list)
		tau, p_value = stats.kendalltau(prediction_list, exp_list)


		# For plotting test set correlations:
		tuples_result = list(zip(perts_list, exp_list, prediction_list))
		nested_list_result = [ list(elem) for elem in tuples_result ]

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

			with open("./opt_tmp/"+str(iteration)+"_"+str(translated_subject)+"_ALFRESCO_TopPerform_SVM.svm", "wb") as file:
				pickle.dump(svm_estimator, file)

				
		return MAEMAD

	# Bayesian Optimisation to search through hyperparameter space. 
	# Prior parameters were found by manual search and preliminary optimisation loops. 
	# For running just dataset 13x500 calls, optimal hyperparameters from 150 calls were used as prior.
	default_parameters = [1.0, 1.0, 1.0]
	print("###########################################")
	print("Created model, optimising hyperparameters..")
	
	search_result= gp_minimize(func=fitness,
								dimensions=dimensions,
								acq_func='EI', #Expected Improvement.
								n_calls=n_calls,
								x0=default_parameters)


	print("###########################################")
	print("Concluded optimal hyperparameters:")
	print(search_result.x)

	print("###########################################")

	# return skopt object and highest scoring model for this replicate:
	return search_result, model_bucket[-1]



##########################################################
##########################################################
######												######
######				Function calls					######
######												######
##########################################################
##########################################################


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
					writer.writerow(["PCA threshold: "+str(PCA_threshold)])
					writer.writerow(["n replicates: "+str(replicates)])
					writer.writerow(["n models in ensemble: "+str(ensemble_size)])
					writer.writerow(["n calls (BO): "+str(n_calls)])
					writer.writerow(["Started program at: "+time.ctime()])

# loop over input file dict:
for dataset, split in dataset_paths.items():
	bucket_df = pd.DataFrame()
	mae_results_per_fold = [["Subject", "MAE", "Replicate"]]
	MAEtauR_results_per_fold = [["Subject", "Correlation Coefficient", "Dataset", "Correlation metric"]]

	print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
	print(time.ctime())
	print("Working on dataset: "+split)
# construct raw dataset, make sure all values are floats; shuffle rows:
	collection = pd.read_csv(dataset, index_col="Perturbation")
	collection = collection.apply(pd.to_numeric).astype(float).sample(frac=1)

	normalised_dataset, labels = NormaliseDatasets(collection)
	
# reduce features (and sparsity) using PCA:
	preprocessed_data = ReduceFeatures(normalised_dataset, PCA_threshold, split)
	
# Split dataset per target:
	split_datasets = SplitDatasets(preprocessed_data, targets, labels)

# start loop on each protein fold:
	for cross_val_split in split_datasets:
		
		
		subject = cross_val_split[0][1].index[0].split(">")[0]

		translated_subject = TranslateTargetNames(subject)
		print("Working on subject:", translated_subject)
		

	# Collect MAEs for statistics:
		MAEs_per_split = []
		models_per_replicate = []
		for i in range(replicates):
		# run tb-CV:
			# reset MAEMAD startpoint per replicate:
			startpoint_MAEMAD = startpoint_BO
			OptimizeResult, top_performer = RF_TF(cross_val_split, split, i)

			models_per_replicate.append(top_performer)

		# construct, cummin and concatenate results of this replicate to the other replicates in the loop:
			split_columns = { 
							"Dataset" : str(split), 
							"MAE/MAD" : OptimizeResult.func_vals,
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
				writer.writerow(["Finished "+translated_subject+", dataset "+split+", replicate "+str(i+1)+" at "+str(time.ctime())])

		# make ensemble of best models; pick n replicates' top performing models:
		
		models_per_replicate = sorted(models_per_replicate, key=lambda x: x[0])

		ensemble_collection = models_per_replicate[:ensemble_size]
		
		i=1
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
				"opt_tmp/"+opt_replicate+"_"+translated_subject+"_ALFRESCO_TopPerform_SVM.svm",
				"opt_output/model"+str(i)+"_"+split+"_"+translated_subject+"_ALFRESCO_TopPerform_SVM.svm"
				)

			i+=1
		# to keep things clean, remove ./opt_tmp:
		shutil.rmtree("./opt_tmp/")

		# write internal validation MAEMAD value:
		internal_val = pd.DataFrame([result_internal], columns=["val_loss"])
		
		internal_val.to_csv("opt_output/"+str(split)+"_"+str(translated_subject)+"_TopPerformer_internalVal_df.csv")

		# write external validation DF:
		with open("opt_output/"+str(split)+"_"+str(translated_subject)+"_TopPerformer_externalVal_df.csv", "w") as file:
			writer = csv.writer(file)
			writer.writerow(["Perturbation", "Experimental ddGoffset (kcal/mol)", "Predicted ddGoffset (kcal/mol)", "Subject"])
			for row in nested_list_result_external:
				writer.writerow(row + [translated_subject])

	
	MAEs_CV = pd.DataFrame(mae_results_per_fold[1:], columns=mae_results_per_fold[0])
	MAEtauR_CV = pd.DataFrame(MAEtauR_results_per_fold[1:], columns=MAEtauR_results_per_fold[0])
	cumulative_MAEtauR_CV = pd.concat([cumulative_MAEtauR_CV, MAEtauR_CV])


cumulative_MAEtauR_CV.to_csv("output/tb-CV_MAEtauR_outputs.csv", index=False)
cumulative_MAEs.to_csv("output/tbCV_BO_MAE.csv")
print("Success, wrote all files to opt_output/.")



