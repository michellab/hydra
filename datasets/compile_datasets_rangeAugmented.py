import pandas as pd
import csv
import os
import numpy as np
import itertools



augment_size = 10

morphs_targets_dict = {
# # JM Lab datasets:
  '../datasets/input/THROMBIN/morph.in': "THROMBIN",
    '../datasets/input/HSP90_3/morph.in': "HSP90_3",
    '../datasets/input/HSP90_2/morph.in': "HSP90_2",
    '../datasets/input/HSP90_1/morph.in': "HSP90_1",
    '../datasets/input/FXR_1/morph.in': "FXR_1",
    '../datasets/input/FXR_2/morph.in': "FXR_2",
    '../datasets/input/ACK1/morph.in': "ACK1",

# FEP+ datasets:
	'../datasets/input/BACE/morph.in': "BACE",
    '../datasets/input/JNK1/morph.in' : "JNK1",
    '../datasets/input/TYK2/morph.in' : "TYK2",
    '../datasets/input/SCHR_THROMBIN/morph.in' : "SCHR_THROMBIN",
	'../datasets/input/CDK2/morph.in' : "CDK2",
	'../datasets/input/PTP1B/morph.in' : "PTP1B",
    '../datasets/input/MCL1/morph.in' : "MCL1"  
    }

def read_morph_file(MorphFilePath):
    # read in morphfile:
    with open(MorphFilePath, 'rt') as morph_file:

    # read morph_pairs for cleaning
        block = list(itertools.takewhile(lambda x: "[protein]" not in x,
            itertools.dropwhile(lambda x: "morph_pairs" not in x, morph_file)))

        morph_list = [w.replace("\n", "").replace("\t","").replace(",", ", ") for w in block]
        morph_pairs = "".join(morph_list)
        
    # clean data and return as nested list:
        try:
            first_cleaned = (morph_pairs.replace("morph_pairs","").replace("=","").replace(",","\n"))
        except:
            print("Error in reading morph file, check if the line \"morph_pairs = ...\" is ordered vertically. Exiting..")
            return
        second_cleaned = (first_cleaned.replace(" ", "").replace(">",", "))
        molecule_pairs = second_cleaned.split("\n")
        perturbation_list = []
        for pert in molecule_pairs:
            if len(pert) != 0: 
                pert_pair = pert.split(", ", 1)
                perturbation_list.append([pert_pair[0] + ">" + pert_pair[1]])
        print("Amount of perturbations:",len(perturbation_list))
        print("#####################################")
    
    return perturbation_list


def load_ddGs_FEP(MorphFilePath):
	summary_path = MorphFilePath.replace("morph.in", "ddGs/FEP/summary.csv")
	print(summary_path)
	with open(summary_path, "r") as file:
		reader = csv.reader(file)

		ddGs_FEP = []
		excluded_ddGs = []
		for row in reader:
			
	# per line in file, take only lines starting with a ligand name:
			if len(row) > 0:
				if not row[0].startswith("#"):
					
		# exclude nonsense (i.e. failed) perturbations:
					try:
						if float(row[2]) <= 50 and float(row[3]) <= 1:
							pert_string = row[0] + ">" + row[1]
							pert_value = float(row[2])
							var_value = float(row[3])
							ddGs_FEP.append([pert_string, pert_value, var_value])
						else:
							excluded_ddGs.append(row)
					except:
						print("Something is wrong with the formatting of the summary.csv")
						#return
		print("Excluded " + str(len(excluded_ddGs)) + " nonsense prediction(s).")

		return ddGs_FEP

def load_ddGs_EXP(experi_path, perturbation_list):
	## experimental data extraction ##
	experi_path_ic50 = experi_path.replace("morph.in", "ddGs/EXP/experimental_dGs.csv")
	# open and process experimental dG_exps as dict:
	
	try:
		with open(experi_path_ic50, "r") as csvfile:
			experi = []
			reader = csv.reader(csvfile)
			for row in reader:
				
				experi.append(row)
		experi_dict = { k[0]:float(k[1]) for k in experi[1:] }

		# prepare nested list to map experimental values to:
		perts = perturbation_list
		ligpairs = [ pert[0].split(">") for pert in perts ]


		# map experimental values to perturbations
		ddGs_EXP = []
		for pair in ligpairs:
			ligAdG_exp = experi_dict.get(pair[0])
			ligBdG_exp = experi_dict.get(pair[1])
		# compute experimental ddG_exp (if a member is not in experimental 
		# this is assumed to be a fictive intermediate):
			try:
				dG_exp = float(ligBdG_exp) - float(ligAdG_exp)
			except TypeError:
				dG_exp = "fictive"
			pert = pair[0] + ">" + pair[1]
			try:
				ddGs_EXP.append([pert, round(float(dG_exp), 4)])
			except ValueError:
				ddGs_EXP.append([pert, "fictive"])
	# if input isn't experimental dGs, but ddGs computed from an IC50 file:
	except FileNotFoundError:
		try:
			experi_path_ddG = experi_path.replace("morph.in", "ddGs/EXP/computed_ddGs.csv")
			
			with open(experi_path_ddG, "r") as csvfile:
				experi = []
				reader = csv.reader(csvfile)
				for row in reader:
					if len(row) > 0:	
						experi.append([row[0]+">"+row[1], float(row[2])])
						
			ddGs_EXP = experi
			print("Found experimental file with precomputed ddGs, continuing..")

		except FileNotFoundError:
			print("Could not find IC50 or experimental precomputed ddG file")
			return



	return ddGs_EXP




def compute_ddG_offsets(ddGs_FEP, ddGs_EXP, target):
	ddG_offsets = []
	excluded_perts = []
	tmp_ddG_corr = [["Perturbation", "Predicted ddG (FEP) (kcal/mol)", "Experimental ddG (kcal/mol)"]]

	# Match perturbations between FEP and EXP:
	for prediction in ddGs_FEP:
		for experi in ddGs_EXP:
			if prediction[0] == experi[0]:
				
				try:
	# Compute ddG offset:
					offset = prediction[1] - experi[1]
					ddG_offsets.append([prediction[0], round(offset, 3), prediction[2]]) 
					


					#tmp_ddG_corr.append([prediction[0], prediction[1], experi[1]])
	# Exclude perturbations where the experimental ddG is "fictive":
				except TypeError:
					excluded_perts.append(experi)
	

	#tmp_df = pd.DataFrame(tmp_ddG_corr[1:], columns=tmp_ddG_corr[0])
	#tmp_df.to_csv(target+".csv")
	# import seaborn as sns
	# import matplotlib.pyplot as plt
	# sns.set()
	# sns.scatterplot(data=tmp_df, x="Predicted ddG (FEP) (kcal/mol)", y="Experimental ddG (kcal/mol)")
	# plt.plot([-2, 2], [-2, 2], linewidth=2, color="crimson")
	# plt.show()


	print("Excluded " + str(len(excluded_perts)) + " perturbations for containing fictive ligands (i.e. intermediates).")

	# remove duplicates:
	ddG_offsets_set = set(tuple(x) for x in ddG_offsets)
	ddG_offsets = [ list(x) for x in ddG_offsets_set ]
	print("Computed " + str(len(ddG_offsets)) + " ddG offsets")

	# write to file:
	if not os.path.exists("./ddG_offsets_rangeAugmented_compiled"):
		os.makedirs("./ddG_offsets_rangeAugmented_compiled")

	with open('./ddG_offsets_rangeAugmented_compiled/perts_ddG_offsets_'+target+'.csv', 'w') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["Perturbation", "ddG_offset", "variance"])
		for row in ddG_offsets:
			writer.writerow(row)

	print("Wrote offsets file to \'./ddG_offsets_rangeAugmented_compiled/perts_ddG_offsets_"+target+".csv\'")
	print("#####################################")
	return ddG_offsets

def build_dataset(pFP_path, dFEAT_path, dPLEC_path, offsets_path):
	# Read files and store to DFs:
	pFP = pd.read_csv(pFP_path, index_col=0)
	pFP_APFPs = pFP.drop(["Member_Similarity (Dice)"], axis=1)		# remove unneeded columns
		
		
	dFEAT = pd.read_csv(dFEAT_path, index_col=0)
	dPLEC = pd.read_csv(dPLEC_path, index_col=0)

	# Merge DFs pairwise by index:


	dataset_123 = pd.concat([pFP_APFPs, dFEAT, dPLEC], axis=1)

	# merge dG_exp column to dataset_123, while excluding fictive perturbations and duplicates:
	
	dG_offsets = pd.read_csv(offsets_path, index_col="Perturbation")
	
	
	dataset_123 = pd.merge(dataset_123, dG_offsets, left_index=True,right_index=True)
	all_perts = pFP.index.values.tolist()
	# FEPped = [ pert[0] for pert in ddGs_FEP ]
	# for pert in all_perts:
	# 	if pert not in FEPped:
	# 		print(pert)
	dataset_123 = (dataset_123[~dataset_123.index.duplicated()])
	
	

	print("#####################################")
	
	return dataset_123


def split_dataset(dataset_123_offset):

	# Isolate datasets from collective dataframe:
	ddG_offset_column = dataset_123_offset.loc[:, "ddG_offset"]
	variance_column = dataset_123_offset.loc[:, "variance"]
	ddG_offset_column = pd.merge(pd.DataFrame(ddG_offset_column), pd.DataFrame(variance_column),
			left_index=True,right_index=True)

	pFP_columns = dataset_123_offset.loc[:, :"pfp255"]
	dFEAT_columns = dataset_123_offset.loc[:, "AATS0Z":"piPC9"]
	dPLEC_columns = dataset_123_offset.loc[:, "plec0": ].drop(["ddG_offset", "variance"], axis=1)

	# Construct individual dataframes and name them, perts are indices, 
	# final columns are ddG_offset and variance, 1 = pFP, 2 = dFEAT, 3 = dPLEC:
	dataset_1 = pd.merge(pd.DataFrame(pFP_columns), pd.DataFrame(ddG_offset_column), 
		left_index=True,right_index=True)
	dataset_1.name = "dataset_1"
	dataset_1.index.name = "Perturbation"

	dataset_2 = pd.merge(pd.DataFrame(dFEAT_columns), pd.DataFrame(ddG_offset_column), 
		left_index=True,right_index=True)
	dataset_2.name = "dataset_2"
	dataset_2.index.name = "Perturbation"

	dataset_3 = pd.merge(pd.DataFrame(dPLEC_columns), pd.DataFrame(ddG_offset_column), 
		left_index=True,right_index=True)
	dataset_3.name = "dataset_3"
	dataset_3.index.name = "Perturbation"

	# construct noise dataframe (to be used as control):
	noise_columns = np.arange(0, 500).tolist()
	noise_columns = ["noise" + str(item) for item in noise_columns]
	
	noise = pd.DataFrame(np.random.randint(0,100, size=(dataset_1.shape[0], 500)),
					 columns=noise_columns,
					 index=dataset_1.index.values)
	dataset_noise = pd.merge(noise, pd.DataFrame(ddG_offset_column), 
		left_index=True,right_index=True)
	dataset_noise.name = "dataset_noise"
	dataset_noise.index.name = "Perturbation"
	
	# Construct paired dataframes and name them:
	dataset_12_nodG_exp = pd.merge(pd.DataFrame(pFP_columns), pd.DataFrame(dFEAT_columns), 
		left_index=True,right_index=True)
	dataset_12 = pd.merge(dataset_12_nodG_exp, pd.DataFrame(ddG_offset_column), 
		left_index=True,right_index=True)
	dataset_12.name = "dataset_12"
	dataset_12.index.name = "Perturbation"

	dataset_13_nodG_exp = pd.merge(pd.DataFrame(pFP_columns), pd.DataFrame(dPLEC_columns), 
		left_index=True,right_index=True)
	dataset_13 = pd.merge(dataset_13_nodG_exp, pd.DataFrame(ddG_offset_column), 
		left_index=True,right_index=True)
	dataset_13.name = "dataset_13"
	dataset_13.index.name = "Perturbation"

	dataset_23_nodG_exp = pd.merge(pd.DataFrame(dFEAT_columns), pd.DataFrame(dPLEC_columns), 
		left_index=True,right_index=True)
	dataset_23 = pd.merge(dataset_23_nodG_exp, pd.DataFrame(ddG_offset_column), 
		left_index=True,right_index=True)
	dataset_23.name = "dataset_23"
	dataset_23.index.name = "Perturbation"

	# Write and return individual files:
	if not os.path.exists("./trainingsets_compiled"):
		os.makedirs("./trainingsets_compiled")
	
	for file in [dataset_1, dataset_2, dataset_3, dataset_12, dataset_13, dataset_23, dataset_noise]:
		print("Writing to \'./trainingsets_compiled/"+file.name+".csv\'..")

		file.to_csv("trainingsets_compiled/"+file.name+".csv", index=True, chunksize=10)
	print("#####################################")

def augment_by_variance(augment_size):
	# read in all trainingset files using a list:
	for feature_set in ["1", "2", "3", "12", "13", "23", "123"]:
		dataset = pd.read_csv("trainingsets_compiled/dataset_"+feature_set+".csv", index_col="Perturbation")
		
		print("Augmenting feature set: ", feature_set)
		column_names = list(dataset.columns[:-1])

		# transform df to nested list for easier row editing:
		dataset_aslist = dataset.values.tolist()

		augmented_list = []
		for row in dataset_aslist:

			features = row[:-2]
			offset = row[-2]
			variance = row[-1]

			# construct a list of ddGoffset values (n=augment_size) based on the variance:
			range_up = np.linspace(offset, offset + variance, num=augment_size/2)
			range_down = np.linspace(offset, offset - variance, num=augment_size/2)
			whole_range = list(range_up) + list(range_down)
			for ddG_offset_val in whole_range:
				augmented_list.append(features + [ddG_offset_val])

		augmented_dataset = pd.DataFrame(augmented_list, columns=column_names)

		# create new index to paste into new DF:
		perts_xx = [augment_size*[pert] for pert in list(dataset.index)]
		perturbation_index = [item for sublist in perts_xx for item in sublist]
		augmented_dataset.index = perturbation_index
		augmented_dataset.index.name = "Perturbation"

		# write out new augmented dataset:
		augmented_dataset.to_csv("trainingsets_compiled/dataset_augmented_"+feature_set+".csv", index=True, chunksize=100)

def main(morphs_targets_dict, augment_size):
	compiled_dataset123 = pd.DataFrame()

	for path, target in morphs_targets_dict.items():
		print(target)

		perturbation_list = read_morph_file(path)
		ddGs_FEP = load_ddGs_FEP(path)
		ddGs_EXP = load_ddGs_EXP(path, perturbation_list)
		
		compute_ddG_offsets(ddGs_FEP, ddGs_EXP, target)

		pFP_path = "../pFP/dFP_output/perts_APFPs_"+target+".csv"
		dFEAT_path = "../dFEAT/dFeatures_output/deltaFeatures_"+target+".csv"
		dPLEC_path = "../dPLEC/dPLECs_output/perts_dPLECs_"+target+".csv"
		offsets_path = "./ddG_offsets_rangeAugmented_compiled/perts_ddG_offsets_"+target+".csv"

		full_dataset = build_dataset(pFP_path, dFEAT_path, dPLEC_path, offsets_path)

	
		compiled_dataset123 = pd.concat([compiled_dataset123, full_dataset])

		# fix a weird bug where columns are shuffled:
		col_names = full_dataset.columns.tolist()
		compiled_dataset123 = compiled_dataset123[col_names]


	print("Built dataset; excluded duplicates. \nThe dimensions of the dataset (123) are " + str(len(compiled_dataset123)) + " rows (i.e. perturbations) and " + str(len(compiled_dataset123.columns)) + " columns (i.e. delta-descriptors).")
	print("Writing to \'./trainingsets_compiled/dataset_123.csv\'..")

	# write to file:
	if not os.path.exists("./trainingsets_compiled"):
		os.makedirs("./trainingsets_compiled")
	compiled_dataset123.index.name = "Perturbation"
	compiled_dataset123.to_csv("trainingsets_compiled/dataset_123.csv", index=True, chunksize=10)
	

	split_dataset(compiled_dataset123)
	
	# augment by variance by loading from files because of memory issues:
	# augment_by_variance(augment_size)


if __name__ == "__main__":
	main(morphs_targets_dict, augment_size)















