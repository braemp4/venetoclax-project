import pickle 
from model_testing import external_validation, test_on_duplicates, test_on_duplicate_moderate_responders
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import seaborn as sns 
import matplotlib.pyplot as plt


with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

results = external_validation(model) 
print(classification_report(results['y_true'], results['y_pred']))
print(f'roc_auc: {roc_auc_score(results['y_true'], results['y_prob'])}')

metrics = {
        'precision': precision_score(results['y_true'], results['y_pred']),
        'recall':  recall_score(results['y_true'], results['y_pred']),
        'f1': f1_score(results['y_true'], results['y_pred']),
        'accuracy': accuracy_score(results['y_true'], results['y_pred']),
        'ROC-AUC': roc_auc_score(results['y_true'], results['y_prob'])
        }



fig, ax = plt.subplots(1, figsize=(8, 5))
sns.barplot(metrics, errorbar=None, lw=1, edgecolor='k', ax=ax)
plt.yticks( [0, 0.1 ,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] )
plt.title("External validation metrics")
plt.savefig('external_validation_metrics_top_model.png')

#duplicate_test = pd.DataFrame(test_on_duplicates(model))
#print(duplicate_test[duplicate_test['duplicate_exvivo'] == True])


#duplicate_test = pd.DataFrame(test_on_duplicate_moderate_responders(model))
#print(duplicate_test[duplicate_test['duplicate_exvivo'] == True])




