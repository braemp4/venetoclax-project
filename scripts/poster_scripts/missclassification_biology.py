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

def get_pred_class(y_pred, y_true):
    if y_pred == 1 and y_true == 1:
        return 'tp'
    elif y_pred == 1 and y_true == 0:
        return "fp"
    elif y_pred == 0 and y_true == 1:
        return "fn"
    elif y_pred == 0 and y_true == 0:
        return "tn"
    else :
        return 'unknown'


pipeline_results_dir = fig_output + "pipeline_output/"
gmt_dir = "../../data/GMT/"
hallmark_gmt = gmt_dir + "h.all.v2026.1.Hs.symbols.gmt"

model_name = "lr_reg"
model_cv_metrics = {}
model_ev_metrics = {}

run_tf_inf = False

datasets = harmonize()
mf_ad = datasets["Moffitt"]
dc_ad = datasets["Decampos"]

cv_results= pd.read_csv(f"{pipeline_results_dir}{model_name}_cv_results.csv").set_index('samples')
ev_results = pd.read_csv(f"{pipeline_results_dir}{model_name}_ev_results.csv").set_index('samples')

mf_ad = mf_ad[cv_results.index]

# pylimma to get degs
mf_ad.obs['y_pred'] = pd.Categorical(cv_results['y_pred'], categories=[0,1], ordered=True)
mf_ad.obs['y_true'] = pd.Categorical(cv_results['y_true'], categories=[0,1], ordered=True)
mf_ad.obs['pred_class'] = pd.Categorical([get_pred_class(mf_ad.obs['y_pred'].loc[y], mf_ad.obs['y_true'].loc[y]) for y in mf_ad.obs_names], categories=['tn', 'tp', 'fp', 'fn'], ordered=True)


comparisons = [('tn', 'tp'), ('tp', 'fp'), ('tp', 'fn'), ('tn', 'fp'), ('tn', 'fn'), ('fp', 'fn')]
# comparison directionality:
# ctrl | exp
# tn | tp 
# tp | fp
# tn | fp 
# tn | fn 
# fp | fn
            

for comparison in comparisons:
    ctrl, exp = comparison
    subset_ad = mf_ad[(mf_ad.obs['pred_class'] == ctrl) | (mf_ad.obs['pred_class'] == exp)]
    print(subset_ad.obs)
    if not subset_ad.obs.index.empty:
        # LIMMA
        pylimma.lm_fit(subset_ad, design="~ pred_class", key="fit")
        pylimma.e_bayes(subset_ad, key="fit")
        results_df = pylimma.top_table(subset_ad, coef=1, key="fit",number=np.inf,sort_by="p")
        results_df['padj'] = results_df["adj_p_value"]
        dc.pl.volcano(data=results_df, x="log_fc", y="padj", top=10, save= fig_output + f'volcano_plot_from_cv_{comparison}.png')


        # GSEAPY
        results_df = results_df.sort_values(by="t")
        pre_res = gp.prerank(rnk=results_df, gene_sets=[hallmark_gmt], threads=8, min_size=5, max_size=1000, outdir=None, seed=4)
        pre_res.res2d = pre_res.res2d.sort_values(by="NES")
        pre_res.res2d["Term"] = (
            pre_res.res2d["Term"]
            .str.replace("h.all.v2026.1.Hs.symbols.gmt__HALLMARK_", "", regex=False)
        )


        pre_res.res2d = pre_res.res2d.sort_values("FDR q-val").drop_duplicates(subset="Term")
        pre_res.res2d = pre_res.res2d.sort_values(by="NES")

        gp.dotplot(pre_res.res2d,column="FDR q-val",cmap=plt.cm.viridis,size=5,figsize=(6, 10),cutoff=0.10,show_ring=True, ofname=fig_output + f"gsea_dotplot_from_cv_{comparison}.png")

        
        if run_tf_inf:
            # DECOUPLER TF INFERENCE
            collectri = dc.op.collectri(organism='human')
            input_for_tf = results_df[['t']].T.rename(index={"t": "sensitive.vs.resistant"})
            tf_acts, tf_padj = dc.mt.ulm(data=input_for_tf, net=collectri)
            msk = (tf_padj.T < 0.05).iloc[:,0]
            tf_acts = tf_acts.loc[:,msk]

            dc.pl.barplot(data=tf_acts, name="sensitive.vs.resistant", save=fig_output + f'TFs_from_cv_{comparison}.png')




dc_ad = dc_ad[ev_results.index]
dc_ad.obs['y_pred'] = pd.Categorical(ev_results['y_pred'], categories=[0,1], ordered=True)
dc_ad.obs['y_true'] = pd.Categorical(ev_results['y_true'], categories=[0,1], ordered=True)
dc_ad.obs['pred_class'] = pd.Categorical([get_pred_class(dc_ad.obs['y_pred'].loc[y], dc_ad.obs['y_true'].loc[y]) for y in dc_ad.obs_names], categories=['tn', 'tp', 'fp', 'fn'], ordered=True)

for comparison in comparisons:
    ctrl, exp = comparison
    subset_ad = dc_ad[(dc_ad.obs['pred_class'] == ctrl) | (dc_ad.obs['pred_class'] == exp)]
    print(subset_ad.obs)
    if not subset_ad.obs.index.empty:

        pylimma.lm_fit(subset_ad, design="~ pred_class", key="fit")
        pylimma.e_bayes(subset_ad, key="fit")
        results_df = pylimma.top_table(subset_ad, coef=1, key="fit",number=np.inf,sort_by="p")
        results_df = results_df.sort_values(by="t")
        results_df['padj'] = results_df['adj_p_value']
        pre_res = gp.prerank(rnk=results_df, gene_sets=[hallmark_gmt], threads=8, min_size=5, max_size=1000, outdir=None, seed=4)
        pre_res.res2d
        pre_res.res2d = pre_res.res2d.sort_values(by="NES")

        pre_res.res2d["Term"] = (
            pre_res.res2d["Term"]
            .str.replace("h.all.v2026.1.Hs.symbols.gmt__HALLMARK_", "", regex=False)
        )


        pre_res.res2d = pre_res.res2d.sort_values("FDR q-val").drop_duplicates(subset="Term")
        pre_res.res2d = pre_res.res2d.sort_values(by="NES")

        gp.dotplot(
            pre_res.res2d,
            column="FDR q-val",
            cmap=plt.cm.viridis,
            size=6,                # dot size scaling
            figsize=(6, 10),
            cutoff=0.1,           # only show terms with FDR q-val <= cutoff
            show_ring=True,
            ofname=fig_output + f"gsea_dotplot_from_ev_{comparison}.png"
        )
        if run_tf_inf:
            input_for_tf = results_df[['t']].T.rename(index={"t": "sensitive.vs.resistant"})
            tf_acts, tf_padj = dc.mt.ulm(data=input_for_tf, net=collectri)

            msk = (tf_padj.T < 0.05).iloc[:,0]
            tf_acts = tf_acts.loc[:,msk]

            plt.title("TF Inference")
            dc.pl.barplot(data=tf_acts, name="sensitive.vs.resistant", save=fig_output + f"TFs_from_ev_{comparison}.png")

