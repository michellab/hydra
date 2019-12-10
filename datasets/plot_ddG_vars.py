import pandas as pd
import csv
import os
import numpy as np
import itertools
import seaborn as sns 
import matplotlib.pyplot as plt 


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
					ddG_offsets.append([prediction[0], round(offset, 3)]) 
					


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
	if not os.path.exists("./ddG_offsets_compiled"):
		os.makedirs("./ddG_offsets_compiled")

	with open('./ddG_offsets_compiled/perts_ddG_offsets_'+target+'.csv', 'w') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["Perturbation", "ddG_offset"])
		for row in ddG_offsets:
			writer.writerow(row)

	print("Wrote offsets file to \'./ddG_offsets_compiled/perts_ddG_offsets_"+target+".csv\'")
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

def build_feats_on_dict(morphs_targets_dict):
	compiled_dataset123 = pd.DataFrame()

	tmp_vars = pd.DataFrame()

	for path, target in morphs_targets_dict.items():
		print(target)

		perturbation_list = read_morph_file(path)
		ddGs_FEP = load_ddGs_FEP(path)
		var_values = [values[2] for values in ddGs_FEP]
		df = pd.DataFrame(var_values, columns=["var"])
		df["target"] = target
		tmp_vars = pd.concat([tmp_vars, df])
	sns.catplot(x="target", y="var", kind="box", data=tmp_vars)
	plt.xlabel("")
	plt.ylabel("Estimation variance")
	plt.xticks(rotation=45, ha="right")
	plt.savefig("variances_boxplot.png", dpi=300)
	plt.show()



dataset_123 = build_feats_on_dict(morphs_targets_dict)













