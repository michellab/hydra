import pandas as pd 
import seaborn as sns 
import csv
import matplotlib.pyplot as plt 
import math

with open("all_ic50s.csv", "r") as file:
	reader = csv.reader(file, delimiter="\t")
	for row in reader:
		print(row[0], row[1], -math.log10(float(row[1])))

## how????		