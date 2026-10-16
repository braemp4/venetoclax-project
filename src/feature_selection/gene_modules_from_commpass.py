import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datasets.Harmonized_Datasets import harmonize
from sklearn.cluster import KMeans
import gseapy as gp
from scipy.stats import pointbiserialr
from statsmodels.stats.multitest import multipletests


def get_features(datasets,output_dir,n_clusters=100, random_state=4, n_init=10, run_enrich=False, cache_feature_transformation=True):
    '''
    Commpass CPM should already be batch corrected!
    We will cluster gene features and correlate modules to t11_14 - take only modules with significant correlation
    500 gene clusters was used in the moffitt paper to find transcriptomic footprints. Im going to try 100 to save computational power.

    return the training adata object with new layer of selected features
    '''
    
    # not sure what info to save yet ... I probably need to save a folder full of results
    # keep this as a folder (gene_modles_k{n_clusters}/)
    gene_module_output = Path(__file__).resolve().parents[2] / "outputs" / "t11_14_gene_modules" / output_dir 

    gmt_folder = Path(__file__).resolve().parents[2] / "data" / "GMT"

    
    gene_modules_df= pd.DataFrame({})


    commpass_ad, moffitt_ad, decampos_ad = datasets["Commpass"], datasets["Moffitt"], datasets["Decampos"]

   
    # check if the folder exists
    print("check if folder exists")

    if gene_module_output.is_file():
        try:
            print("Gene Module file found!")
            gene_modules_df = pd.read_csv(gene_module_output / f"commpass_gmk{n_clusters}.h5ad")

            #gene_modules_df = pd.read_csv(gene_module_output) # rename as commpass_gmk{n_clusters}.h5ad

        except FileNotFoundError as E:
            print("Gene Module File read error!")

    else:
        print("Getting Gene Modules...")
        gene_module_output.mkdir(exist_ok=True)
                
        commpass_ad = commpass_ad[commpass_ad.obs["Trans_1114"].notna()]

        expr_df = commpass_ad.to_df()
 
        # tranpose to make columns = patients 
        expr_df = expr_df.transpose()
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init, init="k-means++").fit(expr_df) # we will get clusters of gene features

        gene_module_mappings = pd.DataFrame({"gene":expr_df.index, "module":kmeans.labels_})
        
        # get enrichment for each gene module
        module_enrichment_results = {}
        if run_enrich:
            for i in range(n_clusters):
                genes_in_module = gene_module_mappings[gene_module_mappings["module"] == i]["gene"]

                # run these gene modules through enrichr to get functional annotations
                if run_enrich:
                    # we should consider overlap and, padj, and score
                    gene_sets = [
                            'c2.all.v2026.1.Hs.symbols.gmt',
                            'c6.all.v2026.1.Hs.symbols.gmt',
                            'c7.all.v2026.1.Hs.symbols.gmt',
                            'c8.all.v2026.1.Hs.symbols.gmt',
                            'h.all.v2026.1.Hs.symbols.gmt'
                            ]

                    gene_sets = [str(gmt_folder / file) for file in gene_sets]
                    enr = gp.enrich(
                            list(genes_in_module),
                            gene_sets = gene_sets,
                            outdir=None, # we could give this a directory actually
                    )
        

        # get modules scores and correlate gene modules to t11;14
        expr_df = expr_df.transpose()
        module_score_matrix = {}
        pvals = [] 
        for i in range(n_clusters):
            module_name = f"module_{i}"
            genes_in_module = gene_module_mappings[gene_module_mappings["module"] == i]["gene"]
            avg_expr = expr_df[genes_in_module].mean(axis=1)
            module_score_matrix[module_name] = avg_expr

            # correlate avg_expr to t11;14
            t11_14_status = [0 if y == "Not Detected" else 1 for y in commpass_ad.obs["Trans_1114"]]
            r, p = pointbiserialr(avg_expr, t11_14_status)
            pvals.append(p)

            # get adjusted p val
        _,padj, _,_,= multipletests(pvals, method="fdr_bh")

        module_score_padj = pd.DataFrame(padj,index=module_score_matrix.keys(), columns=["padj"])
    
        # keep only significantly correlated modules

        module_score_padj = module_score_padj[module_score_padj["padj"] < 0.05]

        gene_module_mappings.index = [f"module_{i}" for i in gene_module_mappings["module"]]
        #print(gene_module_mappings)


        significant_gene_module_mappings = gene_module_mappings.loc[module_score_padj.index]
       

        gene_modules_df = significant_gene_module_mappings
        gene_modules_ad = commpass_ad.copy()
        gene_modules_ad.uns["gene_modles"] = gene_modules_df

        # cache feature engineering results
        if cache_feature_transformation:
            gene_modules_ad.write_h5ad(gene_module_output / f"commpass_gmk{n_clusters}.h5ad")
            #significant_gene_module_mappings.to_csv(gene_module_output)

    
    # Transform datasets
    moffitt_ad = datasets["Moffitt"]
    decampos_ad = datasets["Decampos"]

    gene_module_matrix_layer_cv = {}
    gene_module_matrix_layer_ev = {}
    
    for i in gene_modules_df['module'].unique():

        module_df = gene_modules_df[gene_modules_df['module'] == i]
        #print(module_df['gene'])
        moffitt_ad_subset = moffitt_ad[:,module_df['gene']].copy()
        decampos_ad_subset = decampos_ad[:, module_df['gene']].copy()

        gene_module_matrix_layer_cv[f"module_{i}"] = moffitt_ad_subset.to_df().mean(axis=1)
        gene_module_matrix_layer_ev[f"module_{i}"] = decampos_ad_subset.to_df().mean(axis=1)

    
    gene_module_matrix_layer_cv, gene_module_matrix_layer_ev= pd.DataFrame(gene_module_matrix_layer_cv), pd.DataFrame(gene_module_matrix_layer_ev)

    
    # this layer must be added as unstructured because the number of columns are different
    moffitt_ad.uns['gene_modules'] = gene_module_matrix_layer_cv # save as moffitt_gmtk{n_clusters}.h5ad
    decampos_ad.uns['gene_modules'] = gene_module_matrix_layer_ev # save as decampos_gmtk{n_clusters}.h5ad

    if cache_feature_transformation:
        moffitt_ad.write_h5ad(gene_module_output / f"moffitt_gmktk{n_clusters}.h5ad")
        decampos_ad.write_h5ad(gene_module_output / f"decampos_gmtk{n_clusters}.h5ad")
 
    #return moffitt_ad, decampos_ad
    X_train, X_test = moffitt_ad.uns['gene_modules'], decampos_ad.uns['gene_modules']
    y_train, y_test = moffitt_ad.obs['ABT-199'], decampos_ad.obs['ABT-199']

    return X_train, X_test, y_train, y_test


    
def test():
    moffitt_ad, decampos_ad = get_features()
    print(decampos_ad.uns["gene_modules"])

if __name__ == "__main__":
    test()
