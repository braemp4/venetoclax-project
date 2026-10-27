import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

def binarize_labels(y, method="gmm", random_state=4):
    '''
    this can be its own util
    '''

    if method == "gmm":
        gmm = GaussianMixture(n_components=2, random_state = random_state)
        
        y = np.array(y).reshape(-1,1)

        gmm.fit(y)

        y = gmm.predict(y)
    elif method == 'tertiles':

        first_tertile = np.quantile(y, .33)
        third_tertile = np.quantile(y, .66) 
 
        y = [1 if v < first_tertile else 0 if v > third_tertile else -1 for v in y]
    return y


def binarize_fold_labels(y_test, y_train, method="gmm", random_state=4):

    y_test_class = None
    y_train_class = None

    if method == "gmm":
        gmm = GaussianMixture(n_components=2, random_state=random_state)
        
        y_train = np.array(y_train).reshape(-1,1)
        y_test = np.array(y_test).reshape(-1,1)

        gmm.fit(y_train)

        y_train_class = gmm.predict(y_train)
        y_test_class = gmm.predict(y_test)


    return y_test_class, y_train_class
