import yaml 
from pathlib import Path 
import pandas as pd
import pickle 
import os
import anndata as ad


def track(general_params, feature_engineering_params, cv_results, cv_per_fold_metrics, cv_overall_metrics, ev_results, ev_metrics, ml_model):
    

    general_params = general_params.copy()
    feature_engineering_params = feature_engineering_params.copy()

    feature_engineering_params.pop("random_state", None)
    general_params['feature_engineering'] = feature_engineering_params

    # I FOUND THE BUG - this edits the original params file - when it runs it replace the pointer in memory from an sklearn object to a str
    # the solution is to simply make a copy of the dictionaries
    general_params['ml_model'] = str(general_params["ml_model"])
   
    dir_name = str(general_params["title"] + general_params["date_timestamp"]).strip().replace(" ", "_")
    experiment_output = Path(__file__).resolve().parents[2] / "outputs" / "experiments" / dir_name

    experiment_output.mkdir(parents=False, exist_ok=True)


    # save ml model - same name as experiment directory
    models_output = Path(__file__).resolve().parents[2] / "models"

    pickle_path = dir_name + ".pkl"
    general_params["model_path"] = str(models_output / pickle_path)
    with open(models_output / pickle_path, "wb") as file:
        pickle.dump(ml_model, file, protocol=pickle.HIGHEST_PROTOCOL)
    

    with open(experiment_output / "experiment_parameters.yaml", "w") as file:
        yaml.dump(general_params, file, default_flow_style=False)


      

    pd.DataFrame(cv_results).to_csv(experiment_output / "cv_results.csv")
    pd.DataFrame(cv_per_fold_metrics).to_csv(experiment_output / "cv_per_fold_metrics.csv")
    pd.Series(cv_overall_metrics).to_csv(experiment_output / "cv_overall_metrics.csv")
    pd.DataFrame(ev_results).to_csv(experiment_output / "ev_results.csv")
    pd.Series(ev_metrics).to_csv(experiment_output / "ev_metrics.csv")
     


def fetch_experiment_results(exp_name):
    '''
    fetch experiment results from directory name 
    '''
    
    # get feature engineering results


    experiment_dir = Path(__file__).resolve().parents[2] / "outputs" / "experiments" / exp_name
    
    # prob need to change ft_eng variable names
    with open(experiment_dir / "experiment_parameters.yaml", "r") as file:
        exp_params = yaml.safe_load(file)
    
    # this part that says "gene_module_analysis" is a temp fix and I need to rewrite the logic to get the feature engineering path
    # we can use a dictionary to pair method to output path
    #ft_eng_method = "gene_module_analysis/" 

    ft_eng_method = exp_params['feature_engineering']["method"]
    ft_eng_dir= exp_params["feature_engineering"]["output_dir"]

    #ft_eng_path = Path(__file__).resolve().parents[2] / "outputs" /  ft_eng_method / ft_eng_dir

    ft_eng_path = Path(__file__).resolve().parents[2] / "outputs" /  ft_eng_method / ft_eng_dir
    
    # get feature engineering output
    ft_eng_files = os.listdir(ft_eng_path)
    ft_eng_results = [ad.read_h5ad(f) for f in ft_eng_files]
    
    
    # absolute path that needs to be changed in the future
    with open(exp_params["model_path"], "rb") as file:
        ml_model = pickle.load(file)
        
        
     
    results = {
            "cv_results": pd.read_csv(experiment_dir / "cv_results.csv"),
            "cv_per_fold_metrics": pd.read_csv(experiment_dir / "cv_per_fold_metrics.csv"),
            "cv_overall_metrics": pd.read_csv(experiment_dir / "cv_overall_metrics.csv"),
            "ev_results": pd.read_csv(experiment_dir / "ev_results.csv"),
            "ev_metrics": pd.read_csv(experiment_dir / "ev_metrics.csv")
            # read experimental_parameters.yaml to get folder for feature engineering results
            }


    
    
    return results, ml_model, ft_eng_results
