from figs_output import fig_output 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from datasets.Harmonized_Datasets import harmonize
import pickle
import shap
from matplotlib_venn import venn3, venn3_circles

pipeline_results_dir = fig_output + "pipeline_output/"

model_name = "lr_reg"
model_cv_metrics = {}
model_ev_metrics = {}

datasets = harmonize()
mf_ad = datasets["Moffitt"]
dc_ad = datasets["Decampos"]

cv_results= pd.read_csv(f"{pipeline_results_dir}{model_name}_cv_results.csv").set_index('samples')
ev_results = pd.read_csv(f"{pipeline_results_dir}{model_name}_ev_results.csv").set_index('samples')

mf_ad = mf_ad[cv_results.index]
dc_ad = dc_ad[ev_results.index]

mf_ad.obs['y_pred'] = list(cv_results["y_pred"])
mf_ad.obs['y_true'] = list(cv_results["y_true"])

with open(pipeline_results_dir + "models/lr_reg.pkl", "rb") as f:
    model = pickle.load(f) 




# plot shap top features
training_fts = list(model.feature_names_in_)
X_train, X_test = mf_ad[:,training_fts], dc_ad[:,training_fts]

explainer = shap.LinearExplainer(model, X_train.to_df(), max_samples=111)
shap_values = explainer(X_test.to_df())

shap_values.feature_names
shap_values.values

mean_shap = [np.mean(shap_values.values[i]) for i in range(len(shap_values))]

shap_results = pd.DataFrame([shap_values.feature_names, mean_shap], index=['feature', 'score']).transpose()
shap_results = shap_results[shap_results['score'].notna()]
shap_results['abs_score'] = np.abs(shap_results['score'])
shap_results_df = shap_results.sort_values(by='abs_score', ascending=False)

fig, ax=plt.subplots(1, figsize=(5,5))
sns.barplot(shap_results_df.head(5),y='abs_score', x='feature', ax=ax)
plt.savefig(fig_output + 'top_fts_shap_barplot.png')

# shap native plots 

size=(20,20)
fig, ax = plt.subplots(1, figsize=size)
fig.subplots_adjust(left=0.21)
shap.plots.beeswarm(shap_values, show=False, max_display=10)
plt.savefig(fig_output + "shap_beeswarm.png")


fig, ax = plt.subplots(1, figsize=size)
fig.subplots_adjust(left=0.21)
shap.plots.violin(shap_values, show=False, max_display=10)
plt.savefig(fig_output + "shap_violin.png")

fig, ax = plt.subplots(1, figsize=size)
fig.subplots_adjust(left=0.21)
shap.plots.beeswarm(shap_values.abs, show=False)
plt.savefig(fig_output + "shap_beeswarm_abs.png")

fig, ax = plt.subplots(1, figsize=size)
fig.subplots_adjust(left=0.21)
shap.plots.bar(shap_values, show=False)
plt.savefig(fig_output + "shap_native_bar.png")

fig, ax = plt.subplots(1, figsize=size)
fig.subplots_adjust(left=0.21)
shap.plots.heatmap(shap_values, show=False)
plt.savefig(fig_output + "shap_heatmap.png")

