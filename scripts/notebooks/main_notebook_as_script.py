# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest

from scipy.stats import ttest_ind

from datasets.Harmonized_Datasets import harmonize

import pylimma 
import decoupler as dc
import gseapy as gp

import starbars

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats



# %%
# CONSTANTS
PROBABILITY_CUTOFF = (0.25, 0.75) # inclusive
RANDOM_STATE=4 
N_SPLITS=4
DESEQ_FEATURE_SELECTION = False

# %%
# HELPER FUNCS
def cross_val_classification(X,y_ad, model,n_splits=N_SPLITS, random_state=RANDOM_STATE, verbose=True):
   

    results = {"samples": [],
                'wes_id': [],
               "y_true": [],
               "y_pred": [],
               "y_prob": [],
               }

    fold_metrics = {
            "n_fold": [],
            "precision_score": [],
            "recall_score": [], 
            "f1_score": [],
  
            "accuracy": [],
            "roc_auc": [],
            }
    
    y = y_ad['venetoclax_response']

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_fold= 0
    for i, (train_index, test_index) in enumerate(skf.split(X,y)):
        if verbose:
            print(f"CV FOLD: {n_fold + 1}")

        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]


        # add ground truth values
        results["y_true"] += list(y_test)
        
        # get test samples
        test_samples = X_test.index
        results["samples"] +=list(test_samples)
        wes_ids = y_ad.loc[test_samples]['WES_SampleID']
        results['wes_id'] += list(wes_ids)
       
        

        #sele_k = SelectKBest(k=100)

        #top_features = sele_k.fit(X_train, y_train).get_feature_names_out()

        #X_train = X_train[top_features]
        #X_test = X_test[top_features]

        # Scale the training data
        scl = StandardScaler()
        X_train = scl.fit_transform(X_train)
        X_test = scl.transform(X_test)

        # train model
        model.fit(X_train, y_train)

        # get fold predictions
        y_pred = model.predict(X_test)
        results["y_pred"] +=list(y_pred)

        # get fold probas
        y_prob = model.predict_proba(X_test)[:,1]
        results["y_prob"] +=list(y_prob)

        # record fold metrics
        n_fold +=1
        fold_metrics["n_fold"].append(n_fold)
        fold_metrics["precision_score"].append(precision_score(y_test, y_pred))
        fold_metrics["recall_score"].append(recall_score(y_test, y_pred))
        fold_metrics["f1_score"].append(f1_score(y_test, y_pred))
        fold_metrics["accuracy"].append(accuracy_score(y_test,y_pred))
        fold_metrics["roc_auc"].append(roc_auc_score(y_test, y_prob))

    if verbose:
        print(pd.DataFrame(fold_metrics))

   
    return results, fold_metrics

def get_bootstrap_metrics(results_df, alpha=0.05, bootstraps=10000, verbose=True):
    results_df = results_df.copy().set_index('samples')

    
    metric_statistics= {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    bootstrap_results = {
        'accuracy': (),
        'precision': (),
        'recall': (),
        'f1': (),
        'roc_auc': (),
    }

    # get bootstrap statistics
    for i in range(bootstraps):
        res = resample(results_df)

        y_pred = res['y_pred']
        y_prob = res['y_prob']
        y_true = res['y_true']

        metric_statistics['accuracy'].append(accuracy_score(y_true, y_pred))
        metric_statistics['precision'].append(precision_score(y_true, y_pred))
        metric_statistics['recall'].append(recall_score(y_true, y_pred))
        metric_statistics['f1'].append(f1_score(y_true, y_pred))
        metric_statistics['roc_auc'].append(roc_auc_score(y_true, y_prob))

    # get confidence intervals
    for key in metric_statistics.keys():
        stat = metric_statistics[key]
        stat = sorted(stat)
        
        lower = np.percentile(stat, (1-alpha) / 2)
        upper = np.percentile(stat, alpha+((1-alpha)/2))
        bootstrap_results[key] = (lower, upper)

    if verbose:
        print("ML Classification metric 95% Confidence Infervals: ")
        for key in bootstrap_results.keys():
            lower, upper = bootstrap_results[key]
            print(f"{key.replace('_', ' ').title():<12} [{lower:.3f}, {upper:.3f}]")
            
            
    return bootstrap_results 


# %%
# import data
datasets = harmonize()
mf_ad = datasets['Moffitt']
cm_ad = datasets['Commpass']

# %%
# preprocessing
mf_ad.obs['Disease_Status'].unique()

#exclude SMOL and MGUS
mf_ad = mf_ad[(mf_ad.obs['Disease_Status'] !=  'SMOL') & (mf_ad.obs['Disease_Status'] != "MGUS")]



# %%
# define response groups
sens_cutoff = np.quantile(mf_ad.obs['ABT-199'],1/3)
res_cutoff = np.quantile(mf_ad.obs['ABT-199'], 2/3)

# 1 = sens, 0 = res, -1 moderate
mf_ad.obs['venetoclax_response'] = [1 if y < sens_cutoff else 0 if y > res_cutoff else -1 for y in mf_ad.obs['ABT-199']]
moderate_ad = mf_ad.copy()
moderate_ad = moderate_ad[moderate_ad.obs['venetoclax_response'] == -1]
mf_ad = mf_ad[mf_ad.obs['venetoclax_response'] != -1]


# %%
mf_ad.obs

# %%
# run deseq for feature selection
genes_selected = list(mf_ad.var_names)

if DESEQ_FEATURE_SELECTION:
    cm_ad = cm_ad[cm_ad.obs['Trans_1114'].notna()]
    cm_ad = cm_ad[cm_ad.obs['Country_Skerget'].notna()]

    expr_df = pd.DataFrame(cm_ad.layers['raw_counts'], columns=np.array(cm_ad.var_names))

    

    inference = DefaultInference(n_cpus=8)
        

    dds = DeseqDataSet(counts = expr_df, metadata=pd.DataFrame(cm_ad.obs), design="~Trans_1114", refit_cooks=True, inference=inference) 
    dds.deseq2()

    ds_res = DeseqStats(dds, contrast=["Trans_1114", "Detected", "Not Detected"], inference=inference)
    ds_res.summary()        

    ds_res = ds_res.results_df
   
    
    ds_res= ds_res.sort_values(by="padj", ascending=False)

    dc.pl.volcano(data=ds_res, x="log2FoldChange", y="padj", top=30)
    plt.show()

    ds_res = ds_res[ds_res['padj'] < 0.05]
    ds_res = ds_res[(abs(ds_res['log2FoldChange']) >= 2)]

    genes_selected = ds_res.index

   




# %%
# cross validation 
model = LogisticRegression(random_state=RANDOM_STATE)

results, fold_metrics = cross_val_classification(mf_ad.to_df()[genes_selected], mf_ad.obs,model=model)
results_df = pd.DataFrame(results)

results_df.to_csv('../../outputs/predictions_for_wes_analysis.csv')



# %%
bootstrap_metrics = get_bootstrap_metrics(results_df=results_df, bootstraps=1000)

# %%
# fit model and test on moderate samples
lr_model = LogisticRegression(random_state=RANDOM_STATE)
lr_model.fit(mf_ad.to_df(), mf_ad.obs['venetoclax_response'])

# only combine y_pred,samples and y_prob,samples as secondary for specific tests i.e. diff expr, ols analysis of probabilities 
y_pred = lr_model.predict(moderate_ad.to_df()) # cant get bootstrap metrics from moderate response group because they have no "true" metrics 
y_prob = lr_model.predict_proba(moderate_ad.to_df())[:,1]  
test_samples = moderate_ad.obs_names 
test_results_df = pd.DataFrame({'y_pred': y_pred, 'y_prob': y_prob, 'samples': test_samples})

# %%
# analyze AUC differences 
adata = mf_ad.copy()

#adata = adata[high_conf_results.index]
adata = adata[results_df['samples']]

adata.obs['y_pred'] = list(results_df['y_pred'])
adata.obs['y_prob'] = list(results_df['y_prob'])

fig, ax = plt.subplots(1)
plt.title(f'Venetoclax Response Difference (n={len(adata.obs)})')
sns.boxplot(adata.obs, y='ABT-199', x='y_pred', hue='y_pred')
sns.stripplot(adata.obs, y='ABT-199', x='y_pred',color='k')

test = ttest_ind(adata[adata.obs['y_pred'] == 1].obs['ABT-199'], adata[adata.obs['y_pred'] == 0].obs['ABT-199'])
annotation = (0, 1, f"p val = {test.pvalue:3E}")
starbars.draw_annotation([annotation])

# %%
sensitive_ad = adata[adata.obs['y_pred'] == 1]
resistant_ad = adata[adata.obs['y_pred'] == 0]
# difference in mean expression of key genes
genes_of_interest = ["BCL2", "MCL1","BCL2L1", "BBC3", "NOXA1", "CCND1", 'NFKB1', "RUNX1", "MYC"]

for gene in genes_of_interest:
    test =  ttest_ind(sensitive_ad.to_df()[gene], resistant_ad.to_df()[gene], alternative='two-sided')
    
    fig, ax = plt.subplots(1)
    expr = adata.to_df()
    expr['y_pred'] = adata.obs['y_pred']

    plt.title(gene)
    sns.boxplot(expr, y=gene, x='y_pred', hue='y_pred')
    sns.stripplot(expr, y=gene, x='y_pred',color='k')

    annotation = (0, 1, f"p val = {test.pvalue:3E}")
    starbars.draw_annotation([annotation])




# %%
#adata = mf_ad.copy()
ratios  = ["BCL2/MCL1", "BCL2/BCL2L1"]

adata.obs['BCL2/MCL1'] = adata.to_df()['BCL2'] / adata.to_df()['MCL1']
adata.obs['BCL2/BCL2L1'] = adata.to_df()['BCL2'] / adata.to_df()['BCL2L1']



sensitive_ad = adata[adata.obs['y_pred'] == 1]
resistant_ad = adata[adata.obs['y_pred'] == 0]


for gene in ratios:
    test =  ttest_ind(sensitive_ad.obs[gene], resistant_ad.obs[gene], alternative='two-sided')
    
    fig, ax = plt.subplots(1)
    #expr = adata.to_df()
    #expr['y_pred'] = adata.obs['y_pred']

    plt.title(gene)
    sns.boxplot(adata.obs, y=gene, x='y_pred', hue='y_pred')
    sns.stripplot(adata.obs, y=gene, x='y_pred',color='k')

    annotation = (0, 1, f"p val = {test.pvalue:3E}")
    starbars.draw_annotation([annotation])


# %%
# limma for differential expression analysis
expr = adata.to_df()
design = np.column_stack([np.ones( len(adata.obs['y_pred']) ), np.array(adata.obs['y_pred']) ])

fit = pylimma.lm_fit(expr.transpose(), design)
fit = pylimma.contrasts_fit(fit, contrasts=np.array([[0], [1]]))
fit = pylimma.e_bayes(fit)

results_df = pylimma.top_table(fit,coef=0, key="fit",number=np.inf,sort_by="p")
results_df['padj'] = results_df["adj_p_value"]

dc.pl.volcano(data=results_df, x="log_fc", y="padj", top=30)
plt.show()


# %%
gp.get_library_name()

# %%
import json
# GSEA
plt.clf()
gmt_dir = "../../data/GMT/"
#hallmark_gmt = gmt_dir + "h.all.v2026.1.Hs.symbols.gmt"
gene_set_path = gmt_dir + 'GO_Biological_Process_2026.json'

with open(gene_set_path, 'r') as f:
    gene_set = json.load(f)

results_df = results_df.sort_values(by="t")
pre_res = gp.prerank(rnk=results_df, gene_sets=[gene_set], threads=8, min_size=5, max_size=1000, outdir=None, seed=4)
pre_res.res2d = pre_res.res2d.sort_values(by="NES")
#pre_res.res2d["Term"] = (
#    pre_res.res2d["Term"]
    #.str.replace("h.all.v2026.1.Hs.symbols.gmt__HALLMARK_", "", regex=False)
#)


pre_res.res2d = pre_res.res2d.sort_values("FDR q-val").drop_duplicates(subset="Term")
pre_res.res2d = pre_res.res2d.sort_values(by="NES")



gp.dotplot(pre_res.res2d,column="FDR q-val",cmap="Reds", center=0, title='Predicted Sensitive vs. Resistant', size=5,figsize=(6, 10),cutoff=0.05,show_ring=True,)


# %%
results_df

# %%

collectri = dc.op.collectri(organism='human')
input_for_tf = results_df[['t']].T.rename(index={"t": "sensitive.vs.resistant"})
tf_acts, tf_padj = dc.mt.ulm(data=input_for_tf, net=collectri)
msk = (tf_padj.T < 0.05).iloc[:,0]
tf_acts = tf_acts.loc[:,msk]
fig = dc.pl.barplot(data=tf_acts, name="sensitive.vs.resistant", return_fig = True, top=15, figsize=(5,5))

plt.title('Predicted Sensitive vs. Resistant')
plt.ylabel('Transcription Factor')
plt.xlabel("Enrichment Score")


# %%
tf_acts

# %%


# difference in mean expression of tf genes
genes_of_interest = list(tf_acts.columns)
genes_of_interest = [gene for gene in genes_of_interest]

for gene in genes_of_interest:
    try:
        test =  ttest_ind(sensitive_ad.to_df()[gene], resistant_ad.to_df()[gene], alternative='two-sided')
        
        fig, ax = plt.subplots(1)
        expr = adata.to_df()
        expr['y_pred'] = adata.obs['y_pred']

        plt.title(gene)
        sns.boxplot(expr, y=gene, x='y_pred', hue='y_pred')
        sns.stripplot(expr, y=gene, x='y_pred',color='k')

        annotation = (0, 1, f"p val = {test.pvalue:3E}")
        starbars.draw_annotation([annotation])
    except KeyError as E:
        print(f'missing symbol for TF: {gene}' )





