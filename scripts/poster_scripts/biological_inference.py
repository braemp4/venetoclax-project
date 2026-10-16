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

cv_results= pd.read_csv(f"{pipeline_results_dir}{model_name}_cv_results.csv").set_index('samples')
ev_results = pd.read_csv(f"{pipeline_results_dir}{model_name}_ev_results.csv").set_index('samples')

mf_ad = mf_ad[cv_results.index]
mf_ad.obs['y_pred'] = list(cv_results["y_pred"])
mf_ad.obs['y_true'] = list(cv_results["y_true"])

# pylimma to get degs
mf_ad.obs['y_pred'] = pd.Categorical(cv_results['y_pred'], categories=[0,1], ordered=True)
print(cv_results['y_pred'])

'''
fit = pylimma.lm_fit(mf_ad, design="~ y_pred", key="fit")
pylimma.e_bayes(mf_ad, key="fit")
results_df = pylimma.top_table(mf_ad, coef=1, key="fit",number=np.inf,sort_by="p")
results_df['padj'] = results_df["adj_p_value"]
'''

mf_ad = mf_ad[mf_ad.obs['t11_14'] == 1]

expr = mf_ad.to_df()
design = np.column_stack([np.ones( len(mf_ad.obs['y_pred']) ), np.array(mf_ad.obs['y_pred']) ])

fit = pylimma.lm_fit(expr.transpose(), design)
fit = pylimma.contrasts_fit(fit, contrasts=np.array([[0], [1]]))
fit = pylimma.e_bayes(fit)

results_df = pylimma.top_table(fit,coef=0, key="fit",number=np.inf,sort_by="p")
results_df['padj'] = results_df["adj_p_value"]

dc.pl.volcano(data=results_df, x="log_fc", y="padj", top=30, save= fig_output + 'volcano_plot_t11_14pos_preds.png')
plt.show()


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
pre_res.res2d.to_csv("Moffitt_GSEA_sens_vs_res.csv")
#gp.dotplot(pre_res.res2d,column="FDR q-val",cmap="RdPu" ,size=5,figsize=(6, 10),cutoff=0.25,show_ring=True, ofname=fig_output + "gsea_dotplot_from_cv.png", )
plt.clf()

gp.dotplot(pre_res.res2d,column="FDR q-val",cmap="Reds", center=0, title='Predicted Sensitve vs. Resistant (t11;14+ only)', size=5,figsize=(6, 10),cutoff=0.05,show_ring=True, ofname=fig_output + "sig_gsea_dotplot_t11_14_pos_preds.png", )
plt.clf()

# DECOUPLER TF INFERENCE
if run_tf_inference:
    collectri = dc.op.collectri(organism='human')
    input_for_tf = results_df[['t']].T.rename(index={"t": "sensitive.vs.resistant"})
    tf_acts, tf_padj = dc.mt.ulm(data=input_for_tf, net=collectri)
    msk = (tf_padj.T < 0.05).iloc[:,0]
    tf_acts = tf_acts.loc[:,msk]
    fig = dc.pl.barplot(data=tf_acts, name="sensitive.vs.resistant", return_fig = True, top=15, figsize=(5,5))

    plt.title('Predicted Sensitive vs. Resistant')
    plt.ylabel('Transcription Factor')
    plt.xlabel("Enrichment Score")
    plt.savefig(fig_output + 'TFs_from_cv.png')

'''
# GSEAPY ON EV

dc_ad = dc_ad[ev_results.index]
dc_ad.obs['y_pred'] = pd.Categorical(ev_results['y_pred'], categories=[0,1], ordered=True)

pylimma.lm_fit(dc_ad, design="~ y_pred", key="fit")
pylimma.e_bayes(dc_ad, key="fit")
results_df = pylimma.top_table(dc_ad, coef=1, key="fit",number=np.inf,sort_by="p")
results_df = results_df.sort_values(by="t")
results_df['padj'] = results_df['adj_p_value']
pre_res = gp.prerank(rnk=results_df, gene_sets=[hallmark_gmt], threads=8, min_size=5, max_size=1000, outdir=None, seed=4)


pre_res.res2d["Term"] = (
    pre_res.res2d["Term"]
    .str.replace("h.all.v2026.1.Hs.symbols.gmt__HALLMARK_", "", regex=False)
)



#pre_res.res2d = pre_res.res2d.drop_duplicates(subset='Term')
pre_res.res2d = pre_res.res2d.sort_values("FDR q-val").drop_duplicates(subset="Term")
pre_res.res2d = pre_res.res2d.sort_values(by="NES")

pre_res.res2d.to_csv('external-validation.csv')

gp.dotplot(
    pre_res.res2d,
    column="FDR q-val",
    cmap="Reds",
    size=6,                # dot size scaling
    figsize=(6, 10),
    cutoff=0.25,           # only show terms with FDR q-val <= cutoff
    show_ring=True,
    ofname=fig_output + "gsea_dotplot_from_ev.png"

)
plt.clf()

gp.dotplot(
    pre_res.res2d,
    column="FDR q-val",
    title='HALLMARK GSEA (q < 0.05)',
    cmap="Reds",
    size=6,                # dot size scaling
    figsize=(6, 10),
    cutoff=0.05,           # only show terms with FDR q-val <= cutoff
    show_ring=True,
    ofname=fig_output + "sig_gsea_dotplot_from_ev.png"

)

print(dc_ad.obs['y_pred'])

if run_tf_inference:
    input_for_tf = results_df[['t']].T.rename(index={"t": "sensitive.vs.resistant"})
    tf_acts, tf_padj = dc.mt.ulm(data=input_for_tf, net=collectri)

    msk = (tf_padj.T < 0.05).iloc[:,0]
    tf_acts = tf_acts.loc[:,msk]

    plt.title("TF Inference")
    dc.pl.barplot(data=tf_acts, name="sensitive.vs.resistant", save=fig_output + "TFs_from_ev.png")
'''
