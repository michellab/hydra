import csv
import itertools

import numpy as np
temp = 300

############ Read IC50s' rename: ##############
with open("EXP/ic50s.csv", "r") as file:
	reader = csv.reader(file)
	ic50s = {}
	for row in reader:
		ic50s["ACK1_"+row[0]] = row[1]


############ Read morphs: ##############
with open("../morph.in", "r") as file:
# read morph_pairs for cleaning
    block = list(itertools.takewhile(lambda x: "[protein]" not in x,
        itertools.dropwhile(lambda x: "morph_pairs" not in x, file)))

    morph_list = [w.replace("\n", "").replace("\t","").replace(",", ", ") for w in block]
    morph_pairs = "".join(morph_list)
    
# clean data and return as nested list:
    try:
        first_cleaned = (morph_pairs.replace("morph_pairs","").replace("=","").replace(",","\n"))
    except:
        print("Error in reading morph file, check if the line \"morph_pairs = ...\" is ordered vertically. Exiting..")
        
    second_cleaned = (first_cleaned.replace(" ", "").replace(">",", "))
    molecule_pairs = second_cleaned.split("\n")
    perturbation_list = []
    for pert in molecule_pairs:
        if len(pert) != 0: 
            perturbation_list.append(pert.split(", ", 1))


############ Read IC50s dict per perturbation: ##############

ddGs_list = []
for pert in perturbation_list:

    ic50_member1 = float(ic50s[pert[0]])
    ic50_member2 = float(ic50s[pert[1]])

    # IC50 to ddG equation = Kb * T * ln(ic50(2)/ic50(1))
    ddG = 0.0019872041*temp*np.log(ic50_member2/ic50_member1)
    ddG_pert = [pert[0], pert[1], ddG]
    ddGs_list.append(ddG_pert)

with open("EXP/computed_ddGs.csv", "w") as file:
    writer = csv.writer(file)
    for row in ddGs_list:
        writer.writerow(row)

    



