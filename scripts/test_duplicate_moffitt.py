from datasets.Moffitt import load_moffitt_duplicates_ad


adata = load_moffitt_duplicates_ad()
print(adata.obs)
