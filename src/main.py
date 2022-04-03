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

def create_submission():
    """
    laod saved model and dataframe
    we are using test as the acutals to compare
    model to
    Args:
    returns: print precision, recall and F1 scores
    """

    # read test/submission dataframe
    X_test= np.loadtxt('../data/X_test.txt', dtype=int)
    y_test= np.loadtxt('../data/y_test.txt', dtype=int)

    # read saved model
    model = joblib.load("../model/model_latest_RandomForestClassifier.pkl")

    # Make predictions on test set
    y_true, y_pred = y_test, model.predict(X_test)
    print("Scores for RandomForest on test set")
    print('Precision: %.2f' % precision_score(y_true, y_pred))
    print('Recall: %.2f' % recall_score(y_true, y_pred))
    print('F1 score: %.2f' % f1_score(y_true, y_pred))

if "__main__" == __name__:
    create_submission()
