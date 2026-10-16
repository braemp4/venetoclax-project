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
gmt_dir = "../../data/GMT/"
hallmark_gmt = gmt_dir + "h.all.v2026.1.Hs.symbols.gmt"

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

mf_ad.obs['y_pred'] = cv_results['y_pred']
dc_ad.obs['y_pred'] = ev_results['y_pred']

pathways_of_interest = ['APOPTOSIS', 'MTORC1_SIGNALING', 'TNFA_SIGNALING_VIA_NFKB', 'MYC_TARGETS_V1', "MYC_TARGETS_V2"]
nes = {"t11_14":[], "CV":[], "EV":[]}

# T11;14 ENRICHMENT
mf_ad = mf_ad[mf_ad.obs['t11_14'].notna()]
mf_ad.obs['t11_14'] = mf_ad.obs['t11_14'].astype(int)

# pylimma to get degs
mf_ad.obs['t11_14'] = pd.Categorical(mf_ad.obs['t11_14'], categories=[0,1], ordered=True)

fit = pylimma.lm_fit(mf_ad, design="~ t11_14", key="fit")
pylimma.e_bayes(mf_ad, key="fit")
results_df = pylimma.top_table(mf_ad, coef=1, key="fit",number=np.inf,sort_by="p")
results_df['padj'] = results_df["adj_p_value"]

# GSEAPY
results_df = results_df.sort_values(by="t")
pre_res = gp.prerank(rnk=results_df, gene_sets=[hallmark_gmt], threads=8, min_size=5, max_size=1000, outdir=None, seed=4)
pre_res.res2d = pre_res.res2d.sort_values(by="NES")
pre_res.res2d["Term"] = (
    pre_res.res2d["Term"]
    .str.replace("h.all.v2026.1.Hs.symbols.gmt__HALLMARK_", "", regex=False)
)

#pre_res.res2d = pre_res.res2d.drop_duplicates(subset='Term')
pre_res.res2d = pre_res.res2d.sort_values("FDR q-val").drop_duplicates(subset="Term")
pre_res.res2d = pre_res.res2d.sort_values(by="NES")
t11_14_enrichment = pre_res.res2d

for pathway in pathways_of_interest:
    nes['t11_14'].append(t11_14_enrichment[t11_14_enrichment['Term'] == pathway]["NES"].values)
# MODEL PREDS ENRICHMENT

# pylimma to get degs
mf_ad.obs['y_pred'] = pd.Categorical(mf_ad.obs['y_pred'], categories=[0,1], ordered=True)

fit = pylimma.lm_fit(mf_ad, design="~ y_pred", key="fit")
pylimma.e_bayes(mf_ad, key="fit")
results_df = pylimma.top_table(mf_ad, coef=1, key="fit",number=np.inf,sort_by="p")
results_df['padj'] = results_df["adj_p_value"]

# GSEAPY
results_df = results_df.sort_values(by="t")
pre_res = gp.prerank(rnk=results_df, gene_sets=[hallmark_gmt], threads=8, min_size=5, max_size=1000, outdir=None, seed=4)
pre_res.res2d = pre_res.res2d.sort_values(by="NES")
pre_res.res2d["Term"] = (
    pre_res.res2d["Term"]
    .str.replace("h.all.v2026.1.Hs.symbols.gmt__HALLMARK_", "", regex=False)
)

#pre_res.res2d = pre_res.res2d.drop_duplicates(subset='Term')
pre_res.res2d = pre_res.res2d.sort_values("FDR q-val").drop_duplicates(subset="Term")
pre_res.res2d = pre_res.res2d.sort_values(by="NES")
cv_enrichment = pre_res.res2d

for pathway in pathways_of_interest:
    nes['CV'].append(cv_enrichment[cv_enrichment['Term'] == pathway]["NES"].values)


# pylimma to get degs
dc_ad.obs['y_pred'] = pd.Categorical(dc_ad.obs['y_pred'], categories=[0,1], ordered=True)

fit = pylimma.lm_fit(dc_ad, design="~ y_pred", key="fit")
pylimma.e_bayes(dc_ad, key="fit")
results_df = pylimma.top_table(dc_ad, coef=1, key="fit",number=np.inf,sort_by="p")
results_df['padj'] = results_df["adj_p_value"]

# GSEAPY
results_df = results_df.sort_values(by="t")
pre_res = gp.prerank(rnk=results_df, gene_sets=[hallmark_gmt], threads=8, min_size=5, max_size=1000, outdir=None, seed=4)
pre_res.res2d = pre_res.res2d.sort_values(by="NES")
pre_res.res2d["Term"] = (
    pre_res.res2d["Term"]
    .str.replace("h.all.v2026.1.Hs.symbols.gmt__HALLMARK_", "", regex=False)
)

#pre_res.res2d = pre_res.res2d.drop_duplicates(subset='Term')
pre_res.res2d = pre_res.res2d.sort_values("FDR q-val").drop_duplicates(subset="Term")
pre_res.res2d = pre_res.res2d.sort_values(by="NES")
ev_enrichment = pre_res.res2d

for pathway in pathways_of_interest:
    nes['EV'].append(ev_enrichment[ev_enrichment['Term'] == pathway]["NES"].values)

results_df = pd.DataFrame(nes, index=pathways_of_interest)
results_df.to_csv(pipeline_results_dir + "nes_differences.csv")
print(results_df)

