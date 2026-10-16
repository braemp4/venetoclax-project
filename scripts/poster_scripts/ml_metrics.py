from figs_output import fig_output 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay

pipeline_results_dir = fig_output + "pipeline_output/"

model_names = ["lr_reg", "rf_model", "ada_boost", "xgboost"]
metrics = ["Precision", "Recall", "F1", "Accuracy", "ROC AUC"]

model_cv_metrics = {}
model_ev_metrics = {}

figcv, axcv = plt.subplots(1, figsize=(5,5))
figev, axev = plt.subplots(1, figsize=(5,5))

axcv.plot([0,1], [0,1], linestyle="--", label='Random (AUC = 0.5)', c='grey', alpha=0.5)
axev.plot([0,1], [0,1], linestyle="--", label='Random (AUC = 0.5)', c='grey', alpha=0.5)


colors = sns.color_palette()[:4]

for model_name, color in zip(model_names, colors):
    cv_metrics = pd.read_csv(f"{pipeline_results_dir}{model_name}_cv_metrics.csv")
    ev_metrics = pd.read_csv(f"{pipeline_results_dir}{model_name}_ev_metrics.csv")
    cv_results= pd.read_csv(f"{pipeline_results_dir}{model_name}_cv_results.csv")
    ev_results = pd.read_csv(f"{pipeline_results_dir}{model_name}_ev_results.csv")
    
    model_cv_metrics[model_name] = list(cv_metrics['0'])
    model_ev_metrics[model_name]= list(ev_metrics['0'])

    # plot auc curve cv
    
    if model_name == 'lr_reg':
        cv_disp = RocCurveDisplay.from_predictions(cv_results["y_true"], cv_results["y_prob"], name=model_name)
        cv_disp.plot(curve_kwargs=dict(c=color), ax=axcv)

        ev_disp = RocCurveDisplay.from_predictions(ev_results["y_true"], ev_results["y_prob"], name=model_name)
        ev_disp.plot(curve_kwargs=dict(c=color), ax=axev)

    # plot auc curve ev

axcv.legend()
axev.legend()
axev.set_xlabel('False Positive Rate')
axev.set_ylabel('True Positive Rate')

figcv.savefig(fig_output + "auc_cv.png")

figev.savefig(fig_output + "auc_ev.png")

model_cv_metrics = pd.DataFrame(model_cv_metrics, index=metrics)
model_cv_metrics.columns = ["LR", "RF", "AB", "XGB"]


model_ev_metrics = pd.DataFrame(model_ev_metrics, index=metrics)
model_ev_metrics.columns = ["LR", "RF","AB", "XGB"]

# cv metrics heatmap
fig ,ax= plt.subplots(1, figsize=(5,5))
sns.heatmap(model_cv_metrics,ax=ax, annot=True, fmt=".3f",cmap="Blues")
plt.title("Moffitt - Cross Validation")
plt.savefig(fig_output + "cv_metrics_heatmap.png")
#print(model_cv_metrics)

# ev metrics heatmap
fig ,ax= plt.subplots(1, figsize=(5,5))
sns.heatmap(model_ev_metrics,ax=ax, annot=True, fmt=".3f",cmap="Blues")
plt.title("External Validation")
plt.savefig(fig_output + "ev_metrics_heatmap.png")

