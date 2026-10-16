from figs_output import fig_output 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from datasets.Harmonized_Datasets import harmonize
import pickle
import shap
from matplotlib_venn import venn3, venn3_circles
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import fisher_exact

pipeline_results_dir = fig_output + "pipeline_output/"


datasets = harmonize()
mf_ad = datasets["Moffitt"]
dc_ad = datasets["Decampos"]


model_name = "lr_reg"

cv_results= pd.read_csv(f"{pipeline_results_dir}{model_name}_cv_results.csv")
ev_results = pd.read_csv(f"{pipeline_results_dir}{model_name}_ev_results.csv")

mf_ad = mf_ad[cv_results['samples']]
mf_ad.obs['y_pred'] = list(cv_results["y_pred"])
mf_ad.obs['y_true'] = list(cv_results["y_true"])

with open(pipeline_results_dir + "models/lr_reg.pkl", "rb") as f:
    model = pickle.load(f) 

mf_ad = mf_ad[mf_ad.obs['t11_14'].notna()]
mf_ad.obs['t11_14'] = pd.Categorical(mf_ad.obs['t11_14'].astype(int), categories=[0, 1], ordered=True)

both_correct = 0
fish_only = 0 
model_only = 0
both_wrong = 0

patient_df = pd.DataFrame(mf_ad.obs)

for p in patient_df.index:
    vals = patient_df.loc[p]
    fish_corr = vals["t11_14"] == vals["y_true"]
    model_corr = vals["y_pred"] == vals["y_true"]

    if fish_corr and model_corr:
        both_correct +=1
    elif fish_corr and not model_corr:
        fish_only+=1
    elif model_corr and not fish_corr:
        model_only+=1
    else:
        both_wrong+=1

contingency_table = [[both_correct, fish_only], [model_only, both_wrong]]
contingency_table = pd.DataFrame(contingency_table, columns=["Model Correct", "Model Incorrect"], index=["FISH Correct", "FISH Incorrect"])

result = mcnemar(contingency_table.values, exact=True)
pval = result.pvalue

fig, ax = plt.subplots(1, figsize=(5,5))
sns.heatmap(contingency_table, cmap='Purples', ax=ax, annot=True)
plt.title(f"McNemar's Test p = {pval:.3f}")
plt.savefig(fig_output + "contingency_table_fish_v_model.png")
plt.clf()

conf_matrix = confusion_matrix(mf_ad.obs['y_pred'], mf_ad.obs['t11_14'])

res = fisher_exact(table=conf_matrix)

fig, ax = plt.subplots(1, figsize=(5,5))
sns.heatmap(conf_matrix, cmap='Blues', ax=ax, annot=True)
plt.title(f" Fisher's Exact Test (p = {res.pvalue:.4f})")
plt.xlabel("t11;14 FISH Status")
plt.ylabel("Predicted Sensitivity")

plt.savefig(fig_output + "conf_matrix_fish_v_model.png")
