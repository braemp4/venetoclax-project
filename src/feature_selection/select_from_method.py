from feature_selection import degs_from_commpass, gene_modules_from_commpass, top_ft_anova_from_sens

def select_from(datasets,method, **params):
    '''
    return training set (moffitt_ad) with engineered features.
    Use scripts/list_feature_selection_params.py to view param options for each method

    methods availiable:
    
    {t11_14_degs,
    t11_14_gene_modules,
    top_ft_sens,
    }
    '''
    methods = {
            "t11_14_degs":degs_from_commpass,
            "t11_14_gene_modules":gene_modules_from_commpass,
            "top_ft_sens": top_ft_anova_from_sens,
            } 

    return methods[method].get_features(datasets,**params)
