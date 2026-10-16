import pandas as pd 
import anndata as ad 
import numpy as np 
from pathlib import Path
import mygene

def load_moffitt_ad():
    '''
    Load moffitt anndata object

    If the file is not found we run the data wrangling

    The correct amount of patients with RNAseq and exvivo samples should be 111 if duplicate rnaseq and exvivo ids are consolidated 

    This function will search for ensembl ids for moffitt if the file for them is not found
    '''

    moffitt_path = Path(__file__).resolve().parents[2] / "data" / "Moffitt" #2 means we go 2 levels up from the current directory level. Equivalent to "../../"

    anndata_path = Path(moffitt_path / "AnnData" / "moffitt_annotated.h5ad")
    
    adata = ad.AnnData(pd.DataFrame({"":[]}))

    if anndata_path.is_file():
        print("Moffitt Annotated data found!")
        try: 
            adata = ad.read_h5ad(anndata_path)
        except FileNotFoundError as E:
            print("h5ad read error")

    else:
        print("Creating Annotated data file for moffitt ...")
        #expr_path = "moffitt_log2CPM.csv"
        expr_path = "log2(FPKM+1e-3)_ProteinCodingGenes.csv"
        expr_df = pd.read_csv( moffitt_path / "RNAseq" / expr_path).set_index("GENE").transpose()
        drug_response_df = pd.read_csv(moffitt_path / "ExVivo_Drug_Response" / "1_SingleAgents_Summary_AUCs.csv")

        clinical_df  = pd.read_csv(moffitt_path / "Clinical_Metadata" / "moffit.csv")
       
        # REMOVING DUPLICATE ENTRIES AND ALIGNING INDICES 
        drug_response_df = drug_response_df[drug_response_df["ExVivo_Sample_ID"].notna()].set_index("ExVivo_Sample_ID").loc[~drug_response_df.index.duplicated(keep="first")] # remove duplicates in ex vivo 
        
        
        clinical_df = clinical_df[clinical_df["ExVivo_SampleID"].notna()].set_index("ExVivo_SampleID") # remove na values before setting index to ex vivo ids in clinical metadata
        clinical_df = clinical_df[clinical_df["RNA_SampleID"].notna()] # keep non na values for RNAseq ids within clinical metadata
        clinical_df = clinical_df.loc[~clinical_df.index.duplicated(keep="first")] # remove duplicates in clinical metadata
        

        drug_response_df = drug_response_df.loc[clinical_df.index]
       
        # remove rows without venetoclax AUC values
        drug_response_df = drug_response_df[drug_response_df["ABT-199"].notna()]
        
        clinical_df = clinical_df.loc[drug_response_df.index]
        # merge venetoclax response column with clinical data column 
        clinical_df["ABT-199"] = drug_response_df["ABT-199"]
        

        # match rnaseq ids in clinical metadata to rnaseq count mtatrix

        rna_samples = [idx for idx in clinical_df["RNA_SampleID"] if idx in expr_df.index]
        expr_df = expr_df.loc[rna_samples]
        clinical_df_copy = clinical_df.copy()
        clinical_df_copy["ExVivo_SampleID"] = clinical_df.index

        clinical_df_copy = clinical_df_copy.set_index("RNA_SampleID")
        clinical_df_copy = clinical_df_copy.loc[rna_samples]
        clinical_df_copy = clinical_df_copy.loc[~clinical_df_copy.index.duplicated(keep="first")]

        expr_df = expr_df.loc[clinical_df_copy.index]
        expr_df = expr_df[~expr_df.index.duplicated(keep="first")]




     
        # REMOVE IMMUNOGLOBULIN GENES
        print("Removing Immunoglobulin genes ...")
        non_ig = []
        for col in expr_df.columns:
            if not col.startswith("IGH") or not col.startswith("IGK") or not col.startswith("IGL"):
                non_ig.append(col)
    
        expr_df[non_ig]
        
        #GET ENTREZ GENE ID AND RNA BIOTYPE 
        
        print("Fetching Gene ids ...")
        mg = mygene.MyGeneInfo()
        mg_out = mg.querymany(expr_df.columns, scopes="alias,symbol", fields="type_of_gene,entrezgene,symbol", species="human")
        mg_df = pd.DataFrame(mg_out)
        
        # KEEP ONLY PROTEIN CODING GENES 
        mg_df = mg_df[mg_df["type_of_gene"] == "protein-coding"]

        # DROPPING NAN AND DUPLICATE ENTREZ IDS 
        # this could cause dataset harmonization issues because symbols and entrez ids may be mismatched 
        # A gene with high differences of expression between two datasets may be mismatched

        #mg_df = mg_df[mg_df["entrezgene"].notna()] - commented out for symbols experiment
        #mg_df = mg_df.drop_duplicates(subset=["entrezgene"]) - same as abv


        mg_df = mg_df[mg_df["symbol"].notna()] # added for symbols experiment
        mg_df = mg_df.drop_duplicates(subset=["symbol"]) # same as abv

#        print(mg_df.duplicated())

#        print(mg_df["entrezgene"])
#        print(mg_out)
         
        expr_df = expr_df[mg_df["query"]]

        
     
        # LOADING EVERYTHING INTO ANNDATA
        adata = ad.AnnData(expr_df)
        
        #print(adata.X)
        adata.obs_names = expr_df.index
        #adata.var_names = mg_df["entrezgene"] - commented out for symbols experiment
        adata.var_names = mg_df["symbol"].to_numpy()  # added for symbol experment
        # metadata
        adata.obs["ABT-199"] = clinical_df_copy["ABT-199"]
        adata.obs["ExVivo_SampleID"] = clinical_df_copy['ExVivo_SampleID']
        adata.obs["patientID"] = clinical_df_copy["patientID"]
        adata.obs['t11_14'] = clinical_df_copy["t11_14"]
        adata.obs['WES_SampleID'] = clinical_df_copy['WES_SampleID']
        adata.obs['Disease_Status'] = clinical_df_copy["Disease_status_"]
        
        #print(adata.to_df())
        
        # transcript symbols and rna type 
        adata.var["GeneSymbol"] = mg_df["symbol"].to_numpy()
        adata.var["RNA_Biotype"] = mg_df["type_of_gene"].to_numpy()
    
        adata.write_h5ad(anndata_path) # save file

    return adata

def test():
    ad = load_moffitt_ad()
    print(ad.var)
if __name__ == "__main__":
    test()
