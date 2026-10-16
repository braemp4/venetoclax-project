import inspect
from feature_selection import degs_from_commpass, gene_modules_from_commpass, top_ft_anova_from_sens


modules = [degs_from_commpass, gene_modules_from_commpass, top_ft_anova_from_sens]

for module in modules:
    
    print(module.__name__ + ": " + module.get_features.__name__)
    # print params and defaults
    print(inspect.signature(module.get_features).parameters)
    # print docs
    print(module.get_features.__doc__)
