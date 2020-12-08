import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.metrics import log_loss, make_scorer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# import data
# X_train = pd.read_csv('./data/train/X_train_encoded.csv', index_col=0)
X_train = pd.read_csv('./data/train/X_train_gene_viability_pca.csv', index_col=0)
# y_train = pd.read_csv('./data/train/y_train.csv', index_col=0)
y_train = pd.read_csv("./data/train/y_train.csv", index_col=0, skiprows=0)
# X_test = pd.read_csv('./data/test/X_test_encoded.csv', index_col=0)
X_test = pd.read_csv('./data/test/X_test_gene_viability_pca.csv', index_col=0)
y_test = pd.read_csv('./data/test/y_test.csv', index_col=0)

def grid_search(X_train, y_train):
  model_params = {
    'estimator__n_estimators': [5, 10, 20],
    'estimator__max_depth': [3, 5, 10],
    'estimator__min_samples_split': [4, 7, 10]
  }

  # create random forest classifier model
  rf_model = RandomForestClassifier(random_state=1)
  multi_target_forest = MultiOutputClassifier(rf_model)

  # set up grid search meta-estimator
  # clf = GridSearchCV(rf_model, model_params, cv=3, scoring='neg_log_loss', pre_dispatch='2')
  clf = GridSearchCV(multi_target_forest, param_grid=model_params, cv=3, n_jobs=-1)

  # fit classifier to data
  clf.fit(X_train, y_train)
  print(clf.best_score_, clf.best_estimator_)


grid_search(X_train, y_train)
# print(X_train.head())
# print(y_train.head())