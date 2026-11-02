import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import ttest_ind, wilcoxon
from statsmodels.stats.multitest import multipletests
import pylimma
from pathlib import Path 
import gseapy as gp
import decoupler as dc

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

from sklearn.preprocessing import LabelEncoder


GMT_DIR = Path(__file__).resolve().parents[2] / 'data' / 'GMT'


def map_de_level(a):
    log2FoldChange, nlog10 = a 

    if abs(log2FoldChange) < 1 or nlog10 < 2:
        return "Low DE"
    if abs(log2FoldChange) >= 1 and nlog10 > 2:
        if log2FoldChange >=1:
            return "Overexpressed"
        elif log2FoldChange <=-1:
            return "Underexpressed"
    return "Low DE"

def run_deseq(adata, group_label):

        raw_expr = pd.DataFrame(adata.layers['raw_counts'], columns = np.array(adata.var_names))

        inference = DefaultInference(n_cpus=8)


        dds = DeseqDataSet(counts = raw_expr, metadata=pd.DataFrame(adata.obs), design=f"~{group_label}", refit_cooks=True, inference=inference) 
        dds.deseq2()

         
        ds_res = DeseqStats(dds, contrast=[group_label, 1, 0], inference=inference)

        results_df = ds_res.results_df

        dge_results['gene'] = results_df.index 
        dge_results['t'] = results_df['t']
        dge_results['padj'] = results_df['padj']
        dge_results['log_fc'] = results_df['log2FoldChange']

        return dge_results

def pipeline(adata, dge_method, group_label, gmt_files, genesets_for_gsea, min_tf_activity_targets=5):
    '''
    adata needs a column with the same name as 'group_label'
    group_label classes should be 1 or 0
    'gmt_files' is GSEA input as .gmt format
    'genesets_for_gsea' is GSEA input as dictionary
    
    '''
    expr = adata.to_df()
    dge_results = {'gene':None, 't': None, 'padj':None}

   
    if dge_method == 'limma':
        design = np.column_stack([np.ones( len(adata.obs[group_label]) ), np.array(adata.obs[group_label]) ])
        fit = pylimma.lm_fit(expr.transpose(), design)
        fit = pylimma.contrasts_fit(fit, contrasts=np.array([[0], [1]]))
        fit = pylimma.e_bayes(fit)

        results_df = pylimma.top_table(fit,coef=0, key="fit",number=np.inf,sort_by="p")
        results_df['padj'] = results_df["adj_p_value"]

        dge_results['gene'] = results_df.index 
        dge_results['t'] = results_df['t']
        dge_results['padj'] = results_df['padj']
        dge_results['log_fc'] = results_df['log_fc']
    
    elif dge_method == 'deseq2':
        
        # temp because not all classes will have 1 == Detected but for the most part yes
        # I could also just preprocess the commpass metadata so that Detected == 1

        adata = adata[adata.obs[group_label].notna()]

        y = (adata.obs[group_label] == 'Detected').astype(int)

        adata.obs['group'] = y
        dge_results = run_deseq(adata, group_label='group')



        return dge_results


        
    elif dge_method == 't_test':
        pvals = []
        stat = []
        genes = []
        fc = []
        exp_ad = adata[adata.obs[group_label] == 1]
        ctrl_ad = adata[adata.obs[group_label] == 0]
        

        for gene in expr.columns:
            
            test =  ttest_ind(exp_ad.to_df()[gene], ctrl_ad.to_df()[gene], alternative='two-sided')
            pvals.append(test.pvalue)
            stat.append(test.statistic)
            genes.append(gene)
            fc.append(exp_ad.to_df()[gene].mean() / ctrl_ad.to_df()[gene].mean())

            
        padj = multipletests(pvals)[1]

        dge_results['gene'] = genes
        dge_results['t'] = stat
        dge_results['padj'] = padj
        dge_results['log_fc'] = np.log2(fc)

    elif dge_method == 'wilcoxon':

        pvals = []
        stat = []
        genes = []
        fc = []
        exp_ad = adata[adata.obs[group_label] == 1]
        ctrl_ad = adata[adata.obs[group_label] == 0]
        

        for gene in expr.columns:
            
            test =  wilcoxon(exp_ad.to_df()[gene], ctrl_ad.to_df()[gene], alternative='two-sided')
            pvals.append(test.pvalue)
            stat.append(test.statistic)
            genes.append(gene)
            fc.append(exp_ad.to_df()[gene].mean() / ctrl_ad.to_df()[gene].mean())

            
        padj = multipletests(pvals)[1]

        dge_results['gene'] = genes
        dge_results['t'] = stat
        dge_results['padj'] = padj
        dge_results['log_fc'] = np.log2(fc)

    # GSEA  
    dge_results_df = pd.DataFrame(dge_results).set_index('gene')
   
    # add gmt files to gsea input
    gsea_input = [Path(GMT_DIR / gmt_file) for gmt_file in gmt_files]
    
    # add geneset dictionaries to gsea input
    for key in genesets_for_gsea.keys():
        gsea_input.append({key: genesets_for_gsea[key]})


    dge_results_df = dge_results_df.sort_values(by="t")
    pre_res = gp.prerank(rnk=dge_results_df, gene_sets=gsea_input, threads=8, min_size=5, max_size=1000, outdir=None, seed=4)
    pre_res.res2d = pre_res.res2d.sort_values(by="NES")

    pre_res.res2d = pre_res.res2d.sort_values("FDR q-val").drop_duplicates(subset="Term")
    pre_res.res2d = pre_res.res2d.sort_values(by="NES")

    # TF inference
    collectri = dc.op.collectri(organism='human')
    input_for_tf = dge_results_df[['t']].T.rename(index={"t": "sensitive.vs.resistant"})
    tf_acts, tf_padj = dc.mt.ulm(data=input_for_tf, net=collectri, tmin=min_tf_activity_targets) # set to a minimum
    msk = (tf_padj.T < 0.05).iloc[:,0]
    tf_acts = tf_acts.loc[:,msk]
