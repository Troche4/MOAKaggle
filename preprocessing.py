import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# import data
X_train = pd.read_csv('./data/train/X_train.csv', index_col=0)
X_test = pd.read_csv('./data/test/X_test.csv', index_col=0)


def label_encoding(X_train, X_val, ignore=[]):
    """Function to apply sklearn's LabelEncoder to Train/Test data"""
    # copy data
    X_train_en = X_train.copy()
    X_val_en = X_val.copy()
    # define label encoder
    label_encoder = LabelEncoder()
    # get list of categorical variables
    c = (X_train.dtypes == 'object')
    # if categorical variables in dataset
    if len(c) > 0:
        # get list of categorical features
        features_to_encode = list(c[c].index)
        # ignore columns
        for i in ignore:
            features_to_encode.remove(i)
        for col in features_to_encode:
            X_train_en[col] = label_encoder.fit_transform(X_train[col])
            X_val_en[col] = label_encoder.transform(X_val[col])
    return X_train_en, X_val_en


def pca_processing(X_train, X_val, prefix='g-', n=20, m=50):
    """
    Function to replace column data with the first n Principal Components.
    Function will output a plot of the explained variance of
    the first 50 components.
    Parameters:
        X_train: training data use to fit
        X_val: validation data to be transformed
        prefix: prefix of columns to use for pca
        n: number of principal components to use in data
        m: number of principal components to graph in elbow plot
    """
    # create copies of data
    X_train_pca = X_train.copy()
    X_test_pca = X_val.copy()

    # select gene columns
    genes = []
    for col in X_train.columns:
        if col.startswith(prefix):
            genes.append(col)

    # Drop genes from datasets
    X_train_pca = X_train_pca.drop(columns=genes)
    X_test_pca = X_test_pca.drop(columns=genes)

    # fit PCA on training data using 50 components
    pca = PCA(n_components=m)
    pca.fit(X_train[genes])

    # plot Elbow Plot for first m components
    plt.ylabel("Explained Variance")
    plt.xlabel("# of Components")
    plt.title("PCA Explained Variance")
    plt.ylim(0, max(pca.explained_variance_))
    plt.plot(pca.explained_variance_)
    plt.savefig('./plots/' +str(prefix) + 'PCA_elbow_plot.png')
    plt.close()

    def transform(data):
        """transforms the data and returns the n pcs for each sample"""
    # transform X_train genes and get first n PCs
        gene_pca = pca.transform(data)
        pcs = (data - pca.mean_).dot(pca.components_[0:n].T)
        # rename columns
        column_names = list(pcs.columns)
        for i in range(len(column_names)):
            column_names[i] = str(prefix) + "PC-" + str(column_names[i])
        pcs.columns = column_names
        return pcs

    # create final dataframes
    X_train_pca = X_train_pca.join(transform(X_train[genes]))
    X_test_pca = X_test_pca.join(transform(X_val[genes]))

    return X_train_pca, X_test_pca

"""
Preprocess Data
"""

# encode training and test data
X_train, X_test = label_encoding(X_train, X_test)

# Output encoded datasets to csv
X_train.to_csv('./data/train/X_train_encoded.csv')
X_test.to_csv('./data/test/X_test_encoded.csv')

# Replace gene columns with Principal Components
X_train, X_test = pca_processing(X_train, X_test, prefix='g-', n=20, m=50)

# Export results to csv
X_train.to_csv("./data/train/X_train_gene_pca.csv")
X_test.to_csv("./data/test/X_test_gene_pca.csv")

# Replace cell viability with Principal Components
X_train, X_test = pca_processing(X_train, X_test, prefix='c-', n=2, m=10)

# Export results to csv
X_train.to_csv("./data/train/X_train_gene_viability_pca.csv")
X_test.to_csv("./data/test/X_test_gene_viability_pca.csv")
