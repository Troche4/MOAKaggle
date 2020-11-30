# MOAKaggle
Repository for the MOA Kaggle as a final project for COMP 379/479.


### Data

Data was downloaded from https://www.kaggle.com/c/lish-moa/data

This problem is a multi-label classification problem. The dataset contains gene expression and cell viability data
after treatment with a drug or placebo. The targets are binary Mechanism of Action (MoA) annotations. A drug can have multiple
positive annotations.

This dataset has 875 features and 206 targets. There are 23814 total samples. The dataset was split into 80% training data 
and 20% test data using sklearn's train_test_split method. Code for split is available in `splitdataset.py`.

```python
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
```