import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier

#train data
data = pd.read_csv('data/train/X_train_encoded.csv', index_col=0)
data_pca = pd.read_csv('data/train/X_train_gene_pca.csv', index_col=0)
data_pca_all = pd.read_csv('data/train/X_train_gene_viability_pca.csv', index_col=0)
targets = pd.read_csv('data/train/y_train.csv', index_col=0)


#test data
test_data = pd.read_csv('data/test/X_test_encoded.csv', index_col=0)
test_data_pca = pd.read_csv('data/test/X_test_gene_pca.csv', index_col=0)
test_data_pca_all = pd.read_csv('data/test/X_test_gene_viability_pca.csv', index_col=0)
test_targets = pd.read_csv('data/test/y_test.csv', index_col=0)



# use model
clf = MultiOutputClassifier(estimator=RandomForestClassifier(max_depth=3,
                                                       min_samples_split=4,
                                                       n_estimators=10,
                                                       random_state=1))
svm = SVC(kernel='linear', C=0.001)


dum = MultiOutputClassifier(estimator=DummyClassifier(strategy="stratified"))

labels = ['Encoded Data', 'Gene PCA', 'Gene and Viability PCA']
results_rf = []
results_svm = []
results_dum = []



# run model on encoded data
log_losses= list()
for column in targets.iteritems():
    if len(np.unique(column[1])) > 2:
        # print(np.unique(column[1]))
        svm.fit(data, column[1])
        prediction = svm.predict(X=test_data)
        score = log_loss(test_targets[column[0]], prediction, labels=[0, 1])
    else:
        prediction = np.array([column[1][0] for i in range(len(test_data))])
        score = log_loss(test_targets[column[0]], prediction, labels=[0, 1])
    # print(column[0], "Gene Viability PCA score: ", score)
    log_losses.append(score)
avg_log_loss = sum(log_losses) / len(log_losses)
results_svm.append(avg_log_loss)
print(results_svm)


# run model on data with gene pca
log_losses= list()
for column in targets.iteritems():
    if len(np.unique(column[1])) > 2:
        # print(np.unique(column[1]))
        svm.fit(data_pca, column[1])
        prediction = svm.predict(X=test_data_pca)
        score = log_loss(test_targets[column[0]], prediction, labels=[0, 1])
    else:
        prediction = np.array([column[1][0] for i in range(len(test_data_pca))])
        score = log_loss(test_targets[column[0]], prediction, labels=[0, 1])
    # print(column[0], "Gene Viability PCA score: ", score)
    log_losses.append(score)
avg_log_loss = sum(log_losses) / len(log_losses)
results_svm.append(avg_log_loss)
print(results_svm)

# run model on data with gene and viability pca
log_losses= list()
for column in targets.iteritems():
    if len(np.unique(column[1])) > 2:
        # print(np.unique(column[1]))
        svm.fit(data_pca_all, column[1])
        prediction = svm.predict(X=test_data_pca_all)
        score = log_loss(test_targets[column[0]], prediction, labels=[0, 1])
    else:
        prediction = np.array([column[1][0] for i in range(len(test_data_pca_all))])
        score = log_loss(test_targets[column[0]], prediction, labels=[0, 1])
    # print(column[0], "Gene Viability PCA score: ", score)
    log_losses.append(score)
avg_log_loss = sum(log_losses) / len(log_losses)
results_svm.append(avg_log_loss)
print(results_svm)


# run model on encoded data
clf.fit(data,targets)
pred = clf.predict(test_data)
results_rf.append(log_loss(pred, test_targets))
print(results_rf)

# run model on data with gene pca
clf.fit(data_pca,targets)
pred = clf.predict(test_data_pca)
results_rf.append(log_loss(pred, test_targets))
print(results_rf)

# run model on data with gene and viability pca
clf.fit(data_pca_all,targets)
pred = clf.predict(test_data_pca_all)
results_rf.append(log_loss(pred, test_targets))
print(results_rf)



# run model on encoded data
dum.fit(data,targets)
pred = dum.predict(test_data)
results_dum.append(log_loss(pred, test_targets))
print(results_dum)

# run model on data with gene pca
dum.fit(data_pca,targets)
pred = dum.predict(test_data_pca)
results_dum.append(log_loss(pred, test_targets))
print(results_dum)

# run model on data with gene and viability pca
dum.fit(data_pca_all,targets)
pred = dum.predict(test_data_pca_all)
results_dum.append(log_loss(pred, test_targets))
print(results_dum)


# plot results
plt.bar(labels,results_rf)
plt.ylabel("Log Loss")
plt.title("Log Loss of Random Forest Classifier")
plt.savefig("plots/rf_input_results_test.png")
plt.close()
# plot results
plt.bar(labels,results_svm)
plt.ylabel("Log Loss")
plt.title("Log Loss of SVM Classifier")
plt.savefig("plots/svm_input_results_test.png")
plt.close()
# plot results
plt.bar(labels,results_dum)
plt.ylabel("Log Loss")
plt.title("Log Loss of Dummy Classifier")
plt.savefig("plots/dum_input_results_test.png")
plt.close()

##Plot all Results
x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, results_rf, width, label='Random Forest')
rects2 = ax.bar(x, results_svm, width, label='SVC')
rects3 = ax.bar(x + width, results_dum, width, label='Dummy')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Log Loss')
ax.set_title('Log Loss by Dataset and Model (Test Datasets)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height,4)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    rotation=90,
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.ylim(top=20)
plt.savefig('plots/score_combined_test_data.png')



