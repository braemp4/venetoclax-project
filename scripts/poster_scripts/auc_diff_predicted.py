from figs_output import fig_output 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
from datasets.Harmonized_Datasets import harmonize
from scipy.stats import ttest_ind
import starbars

pipeline_results_dir = fig_output + "pipeline_output/"

model_name = "lr_reg"
model_cv_metrics = {}
model_ev_metrics = {}

datasets = harmonize()
mf_ad = datasets["Moffitt"]
dc_ad = datasets["Decampos"]

cv_results= pd.read_csv(f"{pipeline_results_dir}{model_name}_cv_results.csv")
ev_results = pd.read_csv(f"{pipeline_results_dir}{model_name}_ev_results.csv")


# cv violin w predicted classes - and t-test
mf_ad = mf_ad[cv_results['samples']]
mf_ad.obs['y_pred'] = list(cv_results["y_pred"])
mf_ad.obs['y_true'] = list(cv_results['y_true'])

sens_ad = mf_ad[mf_ad.obs['y_pred'] == 1]
res_ad = mf_ad[mf_ad.obs['y_pred'] == 0]
res = ttest_ind(sens_ad.obs["ABT-199"], res_ad.obs["ABT-199"])
p = res.pvalue
model_pval = p

fig, ax = plt.subplots(1,figsize=(5,5))

plt.title(f"Predicted Sensitive vs. Resistant (t-test p = {p:.3f})")
sns.boxplot(pd.DataFrame(mf_ad.obs),y="ABT-199", x="y_pred",hue="y_pred", ax=ax)
plt.legend()
plt.savefig(fig_output + "violinplot_cv_pred_groups.png")

# t11_14 violine plot
mf_ad = mf_ad[mf_ad.obs['t11_14'].notna()]

sens_ad = mf_ad[mf_ad.obs['t11_14'] == 1]
res_ad = mf_ad[mf_ad.obs['t11_14'] == 0]

res = ttest_ind(sens_ad.obs["ABT-199"], res_ad.obs["ABT-199"])
p = res.pvalue
t11_14_pval = p

fig, ax = plt.subplots(1,figsize=(5,5))

plt.title(f"FISH t11;14 Status (t-test p = {p:.3f})")
sns.violinplot(pd.DataFrame(mf_ad.obs),y="ABT-199", x="t11_14",hue="t11_14", ax=ax)
plt.legend()
plt.savefig(fig_output + "violinplot_moffitt_t11_14.png")


# show violin plot of predicted sens vs res and t11;14 - I want to show the two p values over each side

fig, ax = plt.subplots(1,figsize=(5,5))
mf_ad.obs['t11_14'] = mf_ad.obs['t11_14'].astype(int)
plt.title(f"FISH vs. Model")

mf_ad.obs = mf_ad.obs.sort_values(by='t11_14')

mf_ad.obs['t11_14'] = ['t11;14 +' if y == 1 else 't11;14 -' for y in mf_ad.obs['t11_14']]
sns.boxplot(pd.DataFrame(mf_ad.obs), y='ABT-199', x='t11_14', hue='t11_14', ax=ax, palette={'t11;14 +': 'crimson', "t11;14 -": "lightgray"}, linecolor='k', linewidth=1.5)
starbars.draw_annotation([('t11;14 +', 't11;14 -', f"p = {t11_14_pval:.4f}")])

mf_ad.obs = mf_ad.obs.sort_values(by='y_pred')
mf_ad.obs['y_pred'] = ['Sensitive' if y == 1 else 'Resistant' for y in mf_ad.obs['y_pred']]

sns.boxplot(pd.DataFrame(mf_ad.obs), y='ABT-199', x='y_pred', hue='y_pred', ax=ax, palette={'Sensitive': "crimson", 'Resistant': 'lightgray'}, linecolor='k', linewidth=1.5)

starbars.draw_annotation([('Sensitive', 'Resistant', f"p = {model_pval:.4f}")])

ax.set_xlabel("Label")
plt.legend()
plt.savefig(fig_output + "compare_FISH_model_drug_response.png")


# sens vs t11;14+ 
fig, ax = plt.subplots(1, figsize=(5,5))
sens_ad = mf_ad.copy()
t11_14_ad = mf_ad.copy()

sens_ad = sens_ad[sens_ad['y_pred'] == 1]
t11_14_ad = t11_14_ad[t11_14_ad['t11_14'] == 1]

sns.boxplot(pd.DataFrame(sens_ad.obs), x='y_pred', y='ABT-199')
sns.boxplot(pd.DataFrame(t11_14_ad.obs), x='t11;14', y='ABT-199')





# res vs. t11;14-
