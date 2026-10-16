from figs_output import fig_output 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from datasets.Harmonized_Datasets import harmonize
from scipy.stats import wilcoxon, ttest_ind
from statsmodels.stats.multitest import multipletests
from starbars import draw_annotation


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

mf_ad = mf_ad[mf_ad.obs['t11_14'].notna()]
mf_ad.obs['t11_14'] = mf_ad.obs['t11_14'].astype(int)

mf_ad.obs['t11_14'] = pd.Categorical(mf_ad.obs['t11_14'], categories=[0,1], ordered=True)
mf_ad.obs['y_pred'] = pd.Categorical(mf_ad.obs['y_pred'], categories=[0,1], ordered=True)
dc_ad.obs['y_pred'] = pd.Categorical(dc_ad.obs['y_pred'], categories=[0,1], ordered=True)

adatas = [mf_ad, mf_ad]

labels = ["t11_14", 'y_pred']
genes_of_interest = ["BCL2", "MCL1","BCL2L1", "BBC3", "NOXA1", "CCND1"]
titles = ['t11;14 (Moffitt)', "Model Predictions (Cross-validation)", 'Model Predictions (External Validation)']
mappers = {'y_pred': {0: 'Resistant', 1:"Sensitive"},
           't11_14': {0: "t11;14 -", 1:"t11;14 +"}
           }

results = []
for adata, label, title in zip(adatas, labels, titles):
    label_mapper = mappers[label]
     
    adata = adata[:, genes_of_interest].copy()

    ctrl_ad = adata[adata.obs[label] == 0].copy()
    exp_ad = adata[adata.obs[label] == 1].copy()
    pvals = []
    statistic = []

    

    # BCL2 - alt greater in sens
    bcl2_test = ttest_ind(np.array(ctrl_ad.to_df()['BCL2']), np.array(exp_ad.to_df()['BCL2']), alternative='two-sided')

    pvals.append(bcl2_test.pvalue)
    statistic.append(bcl2_test.statistic)


   
    # MCL1 - alt less in sens 

    mcl1_test = ttest_ind(ctrl_ad.to_df()['MCL1'], exp_ad.to_df()['MCL1'], alternative='two-sided')
    pvals.append(mcl1_test.pvalue)
    statistic.append(mcl1_test.statistic)
    
    # BCL2L1 - alt less in sens

    bcl2l1_test = ttest_ind(ctrl_ad.to_df()['BCL2L1'], exp_ad.to_df()['BCL2L1'], alternative='two-sided')

    pvals.append(bcl2l1_test.pvalue)
    statistic.append(bcl2l1_test.statistic)


    # BBC3 (PUMA) - alt greater in sens

    bbc3_test = ttest_ind(ctrl_ad.to_df()['BBC3'].to_numpy(), exp_ad.to_df()['BBC3'].to_numpy(), alternative='two-sided')

    pvals.append(bbc3_test.pvalue)
    statistic.append(bbc3_test.statistic)

    # NOXA1 - alt greater in sens

    noxa_test = ttest_ind(ctrl_ad.to_df()['NOXA1'], exp_ad.to_df()['NOXA1'], alternative='two-sided')

    pvals.append(noxa_test.pvalue)
    statistic.append(noxa_test.statistic)


    cd1 = ttest_ind(ctrl_ad.to_df()['CCND1'], exp_ad.to_df()['CCND1'], alternative='two-sided')
    pvals.append(cd1.pvalue)
    statistic.append(cd1.statistic)


    df = pd.DataFrame({'pval':pvals,'stat':statistic, 'gene': genes_of_interest}).set_index('gene')
    df['padj'] = multipletests(pvals, method='fdr_bh')[1]
        
    results.append(df)
    print(df)
    
    # boxplot
    
    expr_df= adata.to_df()

    adata.obs[label] = [label_mapper[y] for y in adata.obs[label]]
    expr_df['group'] = adata.obs[label]

    for gene in genes_of_interest:
        c, e, = label_mapper.values()
        annotation = (c, e, f"p adj = {df['padj'].loc[gene]:.4E}")



        fig, ax = plt.subplots(1, figsize=(5,5))
        plt.title(gene + " Expression " + title) 
        cols = ['crimson', 'mediumblue']
        sns.boxplot(expr_df, y=gene, x='group', hue='group', palette=cols, linecolor='k', linewidth=1.5) 

        draw_annotation([annotation])
        plt.savefig(fig_output + title + " " + gene + " " + "mean_expression.png")



ratios = ["BCL2/MCL1", "BCL2/BCL2L1"]
den_ratios = ["MCL1","BCL2L1"]
titles = ['t11;14 (Moffitt)', "Model Predictions (Cross-validation)", 'Model Predictions (External Validation)']
mappers = {'y_pred': {0: 'Resistant', 1:"Sensitive"},
           't11_14': {0: "t11;14 -", 1:"t11;14 +"}
           }

results = []
for adata, label, title in zip(adatas, labels, titles):
    label_mapper = mappers[label]
     
    adata = adata[:, genes_of_interest].copy()

    ctrl_samples = adata[adata.obs[label] == 0].obs_names
    exp_samples = adata[adata.obs[label] == 1].obs_names

    expr_df= adata.to_df()

    pvals = []
    statistic = []
    
    expr_df['BCL2/MCL1'] = expr_df['BCL2'] / expr_df["MCL1"]

    expr_df['BCL2/BCL2L1'] = expr_df['BCL2'] / expr_df["BCL2L1"]
     
    bcl2_mcl1_test = ttest_ind(expr_df['BCL2/MCL1'].loc[ctrl_samples], expr_df['BCL2/MCL1'].loc[exp_samples], alternative='two-sided')

    pvals.append(bcl2_mcl1_test.pvalue)
    statistic.append(bcl2_mcl1_test.statistic)

    bcl2_bcl2l1_test = ttest_ind(expr_df['BCL2/BCL2L1'].loc[ctrl_samples], expr_df['BCL2/BCL2L1'].loc[exp_samples], alternative='two-sided')

    pvals.append(bcl2_bcl2l1_test.pvalue)
    statistic.append(bcl2_bcl2l1_test.statistic)


    df = pd.DataFrame({'pval':pvals,'stat':statistic, 'gene': ratios}).set_index('gene')

    df['padj'] = multipletests(pvals, method='fdr_bh')[1]
        
    results.append(df)


    
    # boxplot

    adata.obs[label] = [label_mapper[y] for y in adata.obs[label]]
    expr_df['group'] = adata.obs[label]

    for gene in ratios:
        c, e, = label_mapper.values()
        annotation = (c, e, f'p adj = {df['padj'].loc[gene]:.4E} ')



        fig, ax = plt.subplots(1, figsize=(5,5))
        plt.title(gene + title) 
        cols = ['crimson', 'mediumblue']
        sns.boxplot(expr_df, y=gene, x='group', hue='group', palette=cols, linecolor='k', linewidth=1.5) 
        draw_annotation([annotation])
        plt.show()

#        plt.savefig(fig_output + title + " " + gene + " " + "mean_expression.png")





