import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier

data = pd.read_csv('data/train/X_train_encoded.csv', index_col=0)
data_pca = pd.read_csv('data/train/X_train_gene_pca.csv', index_col=0)
data_pca_all = pd.read_csv('data/train/X_train_gene_viability_pca.csv', index_col=0)
targets = pd.read_csv('data/train/y_train.csv', index_col=0)

#test data



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
X_train, X_test, y_train, y_test = train_test_split(data, targets, random_state=1, train_size=0.8)

log_losses= list()
for column in y_train.iteritems():
    if len(np.unique(column[1])) > 2:
        # print(np.unique(column[1]))
        svm.fit(X_train, column[1])
        prediction = svm.predict(X=X_test)
        score = log_loss(y_test[column[0]], prediction, labels=[0, 1])
    else:
        prediction = np.array([column[1][0] for i in range(len(X_test))])
        score = log_loss(y_test[column[0]], prediction, labels=[0, 1])
    # print(column[0], "Gene Viability PCA score: ", score)
    log_losses.append(score)
avg_log_loss = sum(log_losses) / len(log_losses)
results_svm.append(avg_log_loss)
print(results_svm)


# run model on data with gene pca
X_train, X_test, y_train, y_test = train_test_split(data_pca, targets, random_state=1, train_size=0.8)

log_losses= list()
for column in y_train.iteritems():
    if len(np.unique(column[1])) > 2:
        # print(np.unique(column[1]))
        svm.fit(X_train, column[1])
        prediction = svm.predict(X=X_test)
        score = log_loss(y_test[column[0]], prediction, labels=[0, 1])
    else:
        prediction = np.array([column[1][0] for i in range(len(X_test))])
        score = log_loss(y_test[column[0]], prediction, labels=[0, 1])
    # print(column[0], "Gene Viability PCA score: ", score)
    log_losses.append(score)
avg_log_loss = sum(log_losses) / len(log_losses)
results_svm.append(avg_log_loss)
print(results_svm)

# run model on data with gene and viability pca
X_train, X_test, y_train, y_test = train_test_split(data_pca_all, targets, random_state=1, train_size=0.8)

log_losses= list()
for column in y_train.iteritems():
    if len(np.unique(column[1])) > 2:
        # print(np.unique(column[1]))
        svm.fit(X_train, column[1])
        prediction = svm.predict(X=X_test)
        score = log_loss(y_test[column[0]], prediction, labels=[0, 1])
    else:
        prediction = np.array([column[1][0] for i in range(len(X_test))])
        score = log_loss(y_test[column[0]], prediction, labels=[0, 1])
    # print(column[0], "Gene Viability PCA score: ", score)
    log_losses.append(score)
avg_log_loss = sum(log_losses) / len(log_losses)
results_svm.append(avg_log_loss)
print(results_svm)


# run model on encoded data
X_train, X_test, y_train, y_test = train_test_split(data, targets, random_state=1, train_size=0.8)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
results_rf.append(log_loss(pred, y_test))
print(results_rf)

# run model on data with gene pca
X_train, X_test, y_train, y_test = train_test_split(data_pca, targets, random_state=1, train_size=0.8)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
results_rf.append(log_loss(pred, y_test))
print(results_rf)

# run model on data with gene and viability pca
X_train, X_test, y_train, y_test = train_test_split(data_pca_all, targets, random_state=1, train_size=0.8)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
results_rf.append(log_loss(pred, y_test))
print(results_rf)



# run model on encoded data
X_train, X_test, y_train, y_test = train_test_split(data, targets, random_state=1, train_size=0.8)
dum.fit(X_train,y_train)
pred = dum.predict(X_test)
results_dum.append(log_loss(pred, y_test))
print(results_dum)

# run model on data with gene pca
X_train, X_test, y_train, y_test = train_test_split(data_pca, targets, random_state=1, train_size=0.8)
dum.fit(X_train,y_train)
pred = dum.predict(X_test)
results_dum.append(log_loss(pred, y_test))
print(results_dum)

# run model on data with gene and viability pca
X_train, X_test, y_train, y_test = train_test_split(data_pca_all, targets, random_state=1, train_size=0.8)
dum.fit(X_train,y_train)
pred = dum.predict(X_test)
results_dum.append(log_loss(pred, y_test))
print(results_dum)


# plot results
plt.bar(labels,results_rf)
plt.ylabel("Log Loss")
plt.title("Log Loss of Random Forest Classifier")
plt.savefig("plots/rf_input_results.png")
plt.close()
# plot results
plt.bar(labels,results_svm)
plt.ylabel("Log Loss")
plt.title("Log Loss of SVM Classifier")
plt.savefig("plots/svm_input_results.png")
plt.close()
# plot results
plt.bar(labels,results_dum)
plt.ylabel("Log Loss")
plt.title("Log Loss of Dummy Classifier")
plt.savefig("plots/dum_input_results.png")
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
ax.set_title('Log Loss by Dataset and Model')
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
plt.savefig('plots/score_combined.png')



