import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import type_of_target

data = pd.read_csv('data/train/X_train_encoded.csv', index_col=0)
data_pca = pd.read_csv('data/train/X_train_gene_pca.csv', index_col=0)
data_pca_all = pd.read_csv('data/train/X_train_gene_viability_pca.csv', index_col=0)
targets = pd.read_csv('data/train/y_train.csv', index_col=0)

X_train, X_test, y_train, y_test = train_test_split(data, targets, random_state=1, train_size=0.8)

# use model
clf = MultiOutputClassifier(estimator=RandomForestClassifier(max_depth=3,
                                                       min_samples_split=4,
                                                       n_estimators=10,
                                                       random_state=1))


labels = ['Encoded Data', 'Gene PCA', 'Gene and Viability PCA']
results = []

# run model on encoded data
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
results.append(log_loss(pred, y_test))

# run model on data with gene pca
X_train, X_test, y_train, y_test = train_test_split(data_pca, targets, random_state=1, train_size=0.8)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
results.append(log_loss(pred, y_test))

# run model on data with gene and viability pca
X_train, X_test, y_train, y_test = train_test_split(data_pca_all, targets, random_state=1, train_size=0.8)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
results.append(log_loss(pred, y_test))

print(results)

# plot results
plt.bar(labels,results)
plt.ylabel("Log Loss")
plt.title("Log Loss of Random Forest Model")
plt.savefig("plots/rf_input_results.png")




