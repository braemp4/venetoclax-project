import pickle 
from model_testing import external_validation, test_on_duplicates, test_on_duplicate_moderate_responders
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd 


with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

results = external_validation(model) 
print(classification_report(results['y_true'], results['y_pred']))
print(f'roc_auc: {roc_auc_score(results['y_true'], results['y_prob'])}')



duplicate_test = pd.DataFrame(test_on_duplicates(model))
print(duplicate_test[duplicate_test['duplicate_exvivo'] == True])


duplicate_test = pd.DataFrame(test_on_duplicate_moderate_responders(model))
print(duplicate_test[duplicate_test['duplicate_exvivo'] == True])




