import pandas as pd 
import anndata as ad 
from pathlib import Path
import numpy as np
import mygene


def load_decampos_ad():
    '''
    Will (probably) be used as the test dataset 
    '''
    
    decampos_path = Path(__file__).resolve().parents[2] / "data" / "DeCampos_Validation"
    
    anndata_path = Path(decampos_path / "decampos_annotated.h5ad") 
    adata = ad.AnnData(pd.DataFrame({"":[]}))

    if anndata_path.is_file():
        print("DeCampos Annotated data found!")

        try:
            adata = ad.read_h5ad(anndata_path)
        except FileNotFoundError as E:
            print("h5ad read error")

    else:
        print("Creating Annotated data file for decampos ...")
        
        # LOAD DATA
        expr_df = pd.read_csv(decampos_path / "fpkm_matrix.csv")
        metadata = pd.read_csv(decampos_path / "sample_metadata.csv")
        
        # set index of expr_df 
        expr_df = expr_df.set_index("Unnamed: 0")
        
        # LOG2 NORMALIZATION OF RNASEQ COUNTS
        expr_df = np.log2(expr_df + 1e-3)
        
        print("Fetching Gene ids ...")
        mg = mygene.MyGeneInfo()
        mg_out = mg.querymany(expr_df.columns, scopes="alias,symbol,reporter,accession,uniprot", fields="type_of_gene,symbol,entrezgene", species="human")        
        
        mg_df = pd.DataFrame(mg_out)
        mg_df = mg_df[mg_df["type_of_gene"] == "protein-coding"]

        mg_df = mg_df[mg_df["symbol"].notna()] 
        mg_df = mg_df.drop_duplicates(subset=["symbol"]) 
        
        expr_df = expr_df[mg_df["query"]]
        
        #expr_df = expr_df[mg_df["query"]]
        
        metadata = metadata.set_index("SRR_ID") 

        # align indices
        metadata = metadata.loc[expr_df.index]


        # LOADING EVERYTHING INTO ANNDATA
        adata = ad.AnnData(expr_df)
        
        #print(adata.X)
        adata.obs_names = expr_df.index
        adata.var_names = mg_df["symbol"].to_numpy()  
        # metadata
        adata.obs["ABT-199"] = metadata['auc']
        #print(adata.to_df())
        
        # transcript symbols and rna type 
        adata.var["GeneSymbol"] = mg_df["symbol"].to_numpy()
        adata.var["RNA_Biotype"] = mg_df["type_of_gene"].to_numpy()
    
        adata.write_h5ad(anndata_path) # save file

    return adata 

    

def test():
    adata = load_decampos_ad()
    print(adata.to_df())

if __name__ == "__main__":
    test()

    
