import pandas as pd
import numpy as np

gtf = pd.read_csv("../data/GTF/GTF_v112.csv")

gene_map_df = gtf[["gene_id", "gene_name"]].copy()

gene_map_df = gene_map_df.drop_duplicates()
gene_map_df = gene_map_df[gene_map_df.notna()]

gene_map_df  = gene_map_df[~gene_map_df.isnull().any(axis=1)]

gene_map_df.to_csv("../data/GTF/ENSEMBL_SYMBOL_MAP.csv")




def map_genes(genes,to_get="ens_id", map_df=gene_map_df):

    if to_get == "ens_id":
        map_df = map_df.set_index("gene_name")
        return map_df.loc[genes]
    elif to_get == "symbol":
        map_df = map_df.set_index('gene_id')
        return map_df.loc[genes]


def test_mapping_genes():
    to_map = ["A1BG", "ZYX", "AAAS"]

    ensembl_ids = map_genes(to_map)
    print(ensembl_ids)

