from figs_output import fig_output 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from datasets.Harmonized_Datasets import harmonize
from matplotlib_venn import venn3, venn3_circles
import pylimma 
import decoupler as dc
import gseapy as gp



pipeline_results_dir = fig_output + "pipeline_output/"
results_df = pd.read_csv(pipeline_results_dir + "nes_differences.csv").set_index("Unnamed: 0")
results_df['pathway'] = results_df.index

fig, ax = plt.subplots(1, figsize=(5,5))
sns.barplot(results_df, y='pathway', x='EV')
plt.savefig(fig_output + 'nes_differences.png')
