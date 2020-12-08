import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.svm import SVC
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV

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

#initialize hyperparameters
hyperparameters = {
    "C":[0.0001], 
    "kernel":['linear'],
    }

#initialize grid searcher and classifier
clf = SVC()
grid_searcher = GridSearchCV(estimator=clf, param_grid=hyperparameters, cv=5) 

#fit, predict, and score for each column in the encoded data
"""
log_losses_encoded = list()
for column in Y_train.iteritems():
    grid_searcher.fit(X=X_train_encoded, y=column[1])
    prediction = grid_searcher.predict(X=X_test_encoded)
    score = log_loss(Y_test[column[0]], prediction)
    print(column[0],"encoded score: ", score)
    log_losses_encoded.append(score)
avg_log_loss_encoded = sum(log_losses_encoded) / len(log_losses_encoded)
print("Avg. log loss value for encoded data: ", avg_log_loss_encoded)
"""

#fit, predict, and score for each column in the gene pca data
log_losses_gene_pca = list()
for column in Y_train.iteritems():
    grid_searcher.fit(X=X_train_gene_pca, y=column[1])
    prediction = grid_searcher.predict(X=X_test_gene_pca)
    score = log_loss(Y_test[column[0]], prediction)
    print(column[0],"Gene PCA score: ", score)
    log_losses_gene_pca.append(score)
avg_log_loss_gene_pca = sum(log_losses_gene_pca) / len(log_losses_gene_pca)
print("Avg. log loss value for gene PCA data: ", avg_log_loss_gene_pca)

#fit, predict, and score for each column in the gene viability / pca data
log_losses_gene_viability_pca = list()
for column in Y_train.iteritems():
    grid_searcher.fit(X=X_train_gene_viability_pca, y=column[1])
    prediction = grid_searcher.predict(X=X_test_gene_viability_pca)
    score = log_loss(Y_test[column[0]], prediction)
    print(column[0],"Gene Viability PCA score: ", score)
    log_losses_gene_viability_pca.append(score)
avg_log_loss_gene_viability_pca = sum(log_losses_gene_viability_pca) / len(log_losses_gene_viability_pca)
print("Avg. log loss value for gene viability PCA data: ", avg_log_loss_gene_viability_pca)

"""
labels = ["Encoded", "Gene PCA", "Gene and Viability PCA"]
results = [avg_log_loss_encoded, avg_log_loss_gene_pca, avg_log_loss_gene_viability_pca]

#plot each avg log_loss value
plt.bar(labels, results)
plt.ylabel("Log Loss")
plt.title("Log Loss of Support Vector Machine model")
plt.savefig("plots/svm_input_results.png")
"""
