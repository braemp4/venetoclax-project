
import pandas as pd 
import anndata as ad 
import numpy as np 
from pathlib import Path
import mygene

def load_commpass_ad():
    '''
    '''

    commpass_path = Path(__file__).resolve().parents[2] / "data" / "CoMMpass" #2 means we go 2 levels up from the current directory level. Equivalent to "../../"

    anndata_path = Path(commpass_path/ "commpass_annotated.h5ad")
    
    adata = ad.AnnData(pd.DataFrame({"":[]}))

    if anndata_path.is_file():
        print("CoMMpass Annotated data found!")
        try: 
            adata = ad.read_h5ad(anndata_path)
        except FileNotFoundError as E:
            print("h5ad read error")

    else:
        print("Creating Annotated data file for CoMMpass ...")
        expr_df = pd.read_csv( commpass_path /  "log2CPM_batch_corrected.csv").set_index("Unnamed: 0")
        raw_expr_df = pd.read_csv(commpass_path / "raw_batch_corrected.csv").set_index("Unnamed: 0")
        #.set_index("gene_name")
        metadata = pd.read_csv(commpass_path/ "patient_clinical_data.csv", sep="\t").set_index("PUBLIC_ID")
        
#        print(expr_df, raw_expr_df)

        print("Fetching Gene ids ...")

        # fetch cpm gene ids 
        mg = mygene.MyGeneInfo()
        mg_out = mg.querymany(expr_df.columns, scopes="alias,symbol", fields="type_of_gene,,symbol", species="human")
        mg_df = pd.DataFrame(mg_out)
        
        # KEEP ONLY PROTEIN CODING GENES 

        mg_df = mg_df[(mg_df['type_of_gene'] == 'protein-coding') & (mg_df["symbol"].notna())].drop_duplicates(subset="symbol", keep="first")

        expr_df = expr_df[mg_df["query"]]
        expr_df.columns = mg_df['symbol']

        expr_df = expr_df.loc[:,~expr_df.columns.duplicated()].copy()
        
        # fetch raw gene ids - convert ens to symbol

        out = mg.querymany(list(raw_expr_df.columns), scopes="ensembl.gene", fields="type_of_gene,symbol", species="human")
        raw_mg_df = pd.DataFrame(out)

        raw_mg_df = raw_mg_df[(raw_mg_df['type_of_gene'] == 'protein-coding') & (raw_mg_df["symbol"].notna())].drop_duplicates(subset="symbol", keep="first")

        raw_expr_df = raw_expr_df[raw_mg_df['query']]
        #raw_expr_df = raw_expr_df.loc[:,~raw_expr_df.columns.duplicated()].copy()
        
        # set to symbol
        raw_expr_df.columns = raw_mg_df["symbol"]

        # get intersection of ids from raw and cpm matrices
        intersected_genes = list(  set(expr_df.columns) & set(raw_expr_df.columns) )



        mg_df, raw_mg_df = mg_df.set_index('symbol'), raw_mg_df.set_index('symbol')

        mg_df, raw_mg_df = mg_df.loc[intersected_genes], raw_mg_df.loc[intersected_genes]


        expr_df = expr_df[intersected_genes]
        raw_expr_df = raw_expr_df[intersected_genes]

        intersected_indices = list(set(expr_df.index) & set(raw_expr_df.index))

        expr_df = expr_df.loc[intersected_indices]
        raw_expr_df = raw_expr_df.loc[intersected_indices]
        
        # ALIGN INDICES WITH METADATA
        intersected_indices = list(set(expr_df.index) & set(metadata.index))
        expr_df = expr_df.loc[intersected_indices]
        raw_expr_df = raw_expr_df.loc[intersected_indices]
        metadata = metadata.loc[intersected_indices]

        
#        print(expr_df)
        # LOADING EVERYTHING INTO ANNDATA
        adata = ad.AnnData(expr_df)
        
        adata.obs_names = expr_df.index
        adata.obs = metadata 
        adata.var_names = expr_df.columns.to_numpy()  

        # add raw counts as layers
        adata.layers["raw_counts"] = raw_expr_df.astype(int)
        
        adata.var["RNA_Biotype"] = mg_df["type_of_gene"].to_numpy()

#        print(adata.X)
    
        adata.write_h5ad(anndata_path) # save file
    return adata

def test():
   ad = load_commpass_ad()
   print(ad.to_df())
if __name__ == "__main__":
    test()
