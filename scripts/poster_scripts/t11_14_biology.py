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

run_tf_inference = True

datasets = harmonize()
mf_ad = datasets["Moffitt"]
dc_ad = datasets["Decampos"]


mf_ad = mf_ad[mf_ad.obs['t11_14'].notna()]
mf_ad.obs['t11_14'] = mf_ad.obs['t11_14'].astype(int)

# pylimma to get degs
mf_ad.obs['t11_14'] = pd.Categorical(mf_ad.obs['t11_14'], categories=[0,1], ordered=True)

fit = pylimma.lm_fit(mf_ad, design="~ t11_14", key="fit")
pylimma.e_bayes(mf_ad, key="fit")
results_df = pylimma.top_table(mf_ad, coef=1, key="fit",number=np.inf,sort_by="p")
results_df['padj'] = results_df["adj_p_value"]
plt.title('t11_14')
dc.pl.volcano(data=results_df, x="log_fc", y="padj", top=10, save= fig_output + 'volcano_plot_t11_14.png')

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

pre_res.res2d.to_csv('t11_14_GSEA.csv')
gp.dotplot(pre_res.res2d,column="FDR q-val",cmap='Reds', size=5,figsize=(6, 10),cutoff=0.05,show_ring=True, ofname=fig_output + "gsea_dotplot_t11_14.png", title='t11;14 Negative vs. t11;14 Positive')
plt.clf()



# DECOUPLER TF INFERENCE
if run_tf_inference:
    collectri = dc.op.collectri(organism='human')
    input_for_tf = results_df[['t']].T.rename(index={"t": "sensitive.vs.resistant"})
    tf_acts, tf_padj = dc.mt.ulm(data=input_for_tf, net=collectri)
    msk = (tf_padj.T < 0.05).iloc[:,0]
    tf_acts = tf_acts.loc[:,msk]

    fig = dc.pl.barplot(data=tf_acts, name="sensitive.vs.resistant", return_fig = True, top=15, figsize=(5,5))

    plt.title('t11;14 Positive vs. Negative')
    plt.ylabel("Transcription Factor")
    plt.xlabel("Enrichment Score")
    plt.savefig(fig_output + "TFs_from_t11_14.png")
    

