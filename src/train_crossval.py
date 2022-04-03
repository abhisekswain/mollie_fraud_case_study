import pandas as pd
import numpy as np
import math
import joblib
import pickle
import datetime
import random

import sklearn
from sklearn import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# for class imbalance
from imblearn.over_sampling import SMOTE


import warnings
warnings.filterwarnings('ignore')

def train_with_cross_val():
    """
    Training with cross validations and gird search
    Args:
    train dataframe
    Returns:
    saved model with best fit parameters
    """
    
    np.random.seed(777)
    # read previously saved feature dataframe
    data_df = pd.read_pickle("../data/df_final.pkl")

    # drop these cols as they are not needed
    cols_to_drop = ['transaction_id', 'batch', 'customer', 'merchant']
    data_df.drop(cols_to_drop, inplace=True, axis=1)

    # dates for splitting dataframe
    train_start_date = "2019-01-01"
    train_end_date = "2019-09-30"
    val_start_date = "2019-10-01"
    val_end_date = "2019-11-30"
    test_start_date = "2019-12-01"
    test_end_date = "2019-12-31"

    data_df = data_df.reset_index().set_index("chck_date")
    data_df.drop('index', axis=1, inplace=True)
    # drop the target variable
    X = data_df.drop(['fraud'], axis=1)

    # The inputs need to be scaled
    scaler = StandardScaler()

    # the target and training columns are defined as follows
    label_col = ["fraud"]
    trainCols = data_df.columns[~data_df.columns.isin(label_col)]

    # split into train and test and scale the training set
    # also apply oversampling to the training set using smote
    X_train = data_df.loc[train_start_date:train_end_date][trainCols]
    y_train = data_df.loc[train_start_date:train_end_date][label_col]
    X_train = scaler.fit_transform(X_train)
    sm=SMOTE(random_state=12345)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)

    # define and scale validation set
    # which will be used to pick the best algorithm
    X_val = data_df.loc[val_start_date:val_end_date][trainCols]
    y_val = data_df.loc[val_start_date:val_end_date][label_col]
    X_val = scaler.fit_transform(X_val)

    # define and scale test set
    # which will be used to report the results
    X_test = data_df.loc[test_start_date:test_end_date][trainCols]
    y_test = data_df.loc[test_start_date:test_end_date][label_col]
    X_test = scaler.fit_transform(X_test)
    np.savetxt('../data/X_test.txt', X_test, fmt='%d')
    np.savetxt('../data/y_test.txt', y_test, fmt='%d')

    # for demo purposes only a few parameters is used 
    # and one model from which produced the best results from a previous run

    tuned_parameters = [[{'n_estimators':[10, 100],'max_features':[ 'log2', 'sqrt'],'criterion' : ['entropy', 'gini'],'max_depth': [100]}]]

    algorithms = [RandomForestClassifier()]
    algorithm_names = ["RandomForestClassifier"]

    for i in range(0, 1):
        print("################   %s   ################" %algorithm_names[i])
        scores = ['f1']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(algorithms[i], tuned_parameters[i], cv=5,
                               scoring='%s' % score)
            clf.fit(X_train, y_train)

            print ("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            joblib.dump(clf, "../model/model_latest_%s.pkl" %algorithm_names[i])
            print ("training done, model saved to file")
            y_true_val, y_pred_val = y_val, clf.predict(X_val)
            print("Scores for %s on validation set:" %algorithm_names[i])
            print('Precision: %.2f' % precision_score(y_true_val, y_pred_val))
            print('Recall: %.2f' % recall_score(y_true_val, y_pred_val))
            print('F1 score: %.2f' % f1_score(y_true_val, y_pred_val))


if __name__ == "__main__":
	train_with_cross_val()
            

     
        


