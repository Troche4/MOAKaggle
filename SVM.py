import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
#from skmultilearn.problem_transform import ClassifierChain

#import preprocessed data
X_train = pd.read_csv('./data/train/x_train.csv', index_col=0)
X_train_encoded = pd.read_csv('./data/train/X_train_encoded.csv', index_col=0)
X_train_gene_pca = pd.read_csv('./data/train/X_train_gene_pca.csv', index_col=0)
X_train_gene_viability_pca = pd.read_csv('./data/train/X_train_gene_viability_pca.csv', index_col=0)

X_test = pd.read_csv('./data/test/x_test.csv', index_col=0)
X_test_encoded = pd.read_csv('./data/test/X_test_encoded.csv', index_col=0)
X_test_gene_pca = pd.read_csv('./data/test/X_test_gene_pca.csv', index_col=0)
X_test_gene_viability_pca = pd.read_csv('./data/test/X_test_gene_viability_pca.csv', index_col=0)

Y_train = pd.read_csv("./data/train/y_train.csv", index_col=0, skiprows=0)
Y_test = pd.read_csv('./data/test/y_test.csv', index_col=0)

#define hyperparameters for C and kernel
hyperparameters = {
    "C": [0.1],
    "kernel": ['rbf']
}
"""
hyperparameters = {
    "C":[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 
    "kernel":['linear', 'poly', 'rbf', 'sigmoid'],
    "gamma":[1, 0.1, 0.01, 0.001, 0.0001]
    }
"""

#initialize pipeline with multi-label classifier
#pipeline_steps = [("scaler", StandardScaler()), ('PCA', PCA()), ('SVM_C', SVC())]
#pipeline = Pipeline(pipeline_steps)

#initialize grid searcher and classifier
clf = SVC()
grid_searcher = GridSearchCV(estimator=clf, param_grid=hyperparameters, cv=2) #set cv = 5 later

#fit to data with each column in Y_train as labels
predictions = np.array([])
for column in Y_train.iteritems():
    grid_searcher.fit(X=X_train_encoded, y=column[1])
    prediction = grid_searcher.predict(X=X_test_encoded)
    print(column[0], "\n\n")
    #print(prediction, "\n\n")
    #print(Y_test[column[0]].to_numpy(), "\n\n")
    np.append(predictions, prediction)
#print(Y_test.to_numpy(),"\n\n")
print(predictions.shape, Y_test.to_numpy().shape)
print(log_loss(Y_test.to_numpy(), predictions), "\n\n")

