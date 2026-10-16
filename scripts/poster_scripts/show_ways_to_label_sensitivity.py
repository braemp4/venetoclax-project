import pandas as pd
import numpy as np
from experiment_tracking.exp_tracker import fetch_experiment_results 
import matplotlib.pyplot as plt 
import seaborn as sns
from datasets.Harmonized_Datasets import harmonize 
from results_path import res_dir 
from figs_output import fig_output
from utils.labelling import binarize_labels
from matplotlib_venn import venn2_circles, venn2
from upsetplot import UpSet, from_contents
from gmm_auc_cutoffs import cutoffs
datasets = harmonize()
dataset_names = ["Moffitt", "Decampos"]
plot_title_suff = ['Moffitt - Cross Validation', "External Validation"]



for name, title_suff in zip(dataset_names, plot_title_suff):
    adata = datasets[name]
    auc = adata.obs['ABT-199']
    # show auc histplot (tertiles)
    first_tertile = np.quantile(auc, .33)
    third_tertile = np.quantile(auc, .66) 
    
    print(first_tertile, third_tertile)
    fig, ax = plt.subplots(1, figsize = (5,5))
    adata.obs["Venetoclax Sensitivity"] = ["Sensitive" if y < first_tertile else "Resistant" if y > third_tertile else "Moderate" for y in auc.values]

    adata.obs["tertiles"] = ["Sensitive" if y < first_tertile else "Resistant" if y > third_tertile else "Moderate" for y in auc.values]
    
    df = pd.DataFrame(adata.obs)
    df = df[df['tertiles'] !='Moderate']
    
    cols = ['crimson', 'lightgray']
    sns.displot(df, x="ABT-199",palette=cols, hue='Venetoclax Sensitivity', kind='kde', multiple="stack")
    
    plt.axvline(first_tertile, ymin=0, ymax=1000, ls="--", label="AUC 1st Tertile", c="b", lw=2,alpha=0.3)
    plt.axvline(third_tertile, ymin=0, ymax=1000, ls="--", label="AUC 3rd Tertile", c="r", lw=2, alpha=0.3) 
    plt.legend()
    #plt.savefig(fig_output + f"{name}_auc_tertiles_histplot.png")

    # show violin plot (tertiles)
    fig, ax = plt.subplots(1, figsize=(5,5))
    sns.violinplot(df, x='Venetoclax Sensitivity', y="ABT-199", hue='Venetoclax Sensitivity', palette=cols, linewidth=2, linecolor='k') 
    plt.title('Tertiles Labels ' + title_suff)
    plt.legend()
    plt.savefig(fig_output + f"{name}_auc_tertiles_violinplot.png")

 
    # show auc histplot (GMM)
    fig, ax = plt.subplots(1, figsize = (5,5))
    adata.obs["Venetoclax Sensitivity"] = binarize_labels(auc)

    adata.obs["Venetoclax Sensitivity"] = ["Sensitive" if y == 1 else "Resistant" for y in adata.obs["Venetoclax Sensitivity"]]

    adata.obs["GMM"] = adata.obs['Venetoclax Sensitivity']
    cols = ['crimson', 'lightgray']
    sns.displot(adata.obs, x="ABT-199",palette=cols, hue='Venetoclax Sensitivity', kind='kde', multiple="stack")
    plt.legend()
    plt.savefig(fig_output + f"{name}_auc_gmm_histplot.png")

    # show violin plot(gmm)

    fig, ax = plt.subplots(1, figsize=(5,5))
    sns.violinplot(adata.obs, x='Venetoclax Sensitivity', y="ABT-199", hue='Venetoclax Sensitivity', palette=cols, linewidth=2, linecolor='k') 

    plt.title("GMM Labels " + title_suff)
    plt.legend()
    plt.savefig(fig_output + f"{name}_auc_gmm_violinplot.png")


    # show violine plot (gmm with cutoff (80-90)) 
    low, high = cutoffs[name]

    adata = adata[(adata.obs['ABT-199'] < low) | (adata.obs["ABT-199"] > high)]


    fig, ax = plt.subplots(1, figsize=(5,5))
    sns.violinplot(adata.obs, x='Venetoclax Sensitivity', y="ABT-199", hue='Venetoclax Sensitivity', palette=cols, linewidth=1.5, linecolor='k') 

    plt.title("GMM Labels (w/ cutoff) " + title_suff)
    plt.legend()
    plt.savefig(fig_output + f"{name}_auc_gmm_cutoff_violinplot.png")
