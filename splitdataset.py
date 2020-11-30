import pandas as pd
from sklearn.model_selection import train_test_split

# read 'train_features' from kaggle.
data_features = pd.read_csv('./data/lish-moa/train_features.csv', index_col=0)
# read 'train_targets_scored'
data_targets = pd.read_csv('./data/lish-moa/train_targets_scored.csv', index_col=0)

# split data into 80% train / test data
X_train, X_test, y_train, y_test = train_test_split(data_features, data_targets, random_state=1, train_size=0.8)

# output datasets to csv
X_train.to_csv('./data/train/X_train.csv')
X_test.to_csv('./data/test/X_test.csv')
y_train.to_csv('./data/train/y_train.csv')
y_test.to_csv('./data/test/y_test.csv')
