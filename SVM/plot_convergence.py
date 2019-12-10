import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import seaborn as sns 
import numpy as np

df = pd.read_csv("output/tbCV_BO_MAE.csv", index_col=[0])


df["Calls"] = df.index
print(df)

noise_dataset = df.loc[df["Dataset"] == "Noise"]
noise_dataset["type"] = "control"

numerical_datasets = df.loc[df["Dataset"] != "Noise"]
#numerical_datasets = df
numerical_datasets = numerical_datasets.astype({"Dataset" : int})
numerical_datasets = numerical_datasets.sort_values("Dataset")
numerical_datasets = numerical_datasets.astype({"Dataset" : int})
numerical_datasets["type"] = "data"

df = pd.concat([noise_dataset, numerical_datasets])
#df = numerical_datasets
df = df.rename(index=str, columns={"Dataset":"Feature set"})

print(df.groupby("Feature set").mean())

sns.set(
	rc={#'figure.figsize':(10,8),
		"lines.linewidth": 2},
	font_scale=1.5
	)

ax = sns.relplot(x="Calls", y="MAE/MAD", 
			#hue="Dataset",
			ci="sd", 
			#row="position",
			col="Subject",
			col_wrap=3,
			hue="Feature set",
			kind="line",
			style="type",
			legend="full",
			palette=[
					'navy', 
					'maroon', 
					'navy', 
					'navy',
					'navy', 
					'navy', 
					'navy',
					'navy'],
			data=df,
			)

leg=ax._legend
leg.set_bbox_to_anchor([0.99, 0.12])



#ax.fig.set_size_inches(46,20)

plt.xlim(0, df["Calls"].max())
plt.ylim(0.5, 1.1)

#ax.fig.tight_layout(h_pad=0.01, w_pad=0.2)


plt.setp(ax._legend.get_title(), fontsize=20)
plt.setp(ax._legend.get_texts(), fontsize=20)  
ax.map(plt.yticks, fontsize=20, color="black")
ax.map(plt.xticks, fontsize=20, color="black")

plt.xticks(fontsize=20)

ax.set_axis_labels("n calls", "Normalised prediction MAE (kcal/ mol)")
ax.set_titles("{col_name}")

plt.savefig("output/convergence_plot_50x40.png")

#plt.show()
