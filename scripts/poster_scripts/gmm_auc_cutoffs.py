cutoffs = {"Moffitt": (75, 90), "Decampos": (0.4,0.7)}

def cutoff_moderate_auc(adata, name):
    low, high = cutoffs[name]
    return adata[(adata.obs["ABT-199"] < low) | (adata.obs["ABT-199"] > high)]

