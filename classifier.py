from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def datasset_load():
    train = pd.read_csv('../archive/sign_mnist_train/sign_mnist_train.csv')
    test = pd.read_csv('../archive/sign_mnist_test/sign_mnist_test.csv')
    label_map = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y', 'Z']

    X_train_ALL = train.values[:, 1:].astype(np.float32())
    Y_train_ALL = train.values[:, 0]
    X_test = test.values[:, 1:].astype(np.float32())
    Y_test = test.values[:, 0]

    # split validation set
    split = StratifiedShuffleSplit(n_splits=1, test_size=2500)
    for train_index, test_index in split.split(X_train_ALL, Y_train_ALL):
        X_train, X_val = X_train_ALL[train_index], X_train_ALL[test_index]
        Y_train, Y_val = Y_train_ALL[train_index], Y_train_ALL[test_index]

    return label_map, X_train, X_val, Y_train, Y_val, X_test, Y_test


def data_transform_standard(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def data_transform_minmax(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def model_fit(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(train_pred, Y_train)
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(test_pred, Y_test)
    return [train_acc, test_acc]


if __name__ == '__main__':
    models_data = {}
    label_map, X_train, X_val, y_train, Y_val, X_test, y_test = datasset_load()
    x_train_standard, x_test_standard = data_transform_standard(X_train, X_test)
    x_train_minmax, x_test_minmax = data_transform_minmax(X_train, X_test)

    # different classifier
    # Naive Bayes
    gnb = GaussianNB()
    models_data['GaussianNB'] = model_fit(gnb, x_train_standard, y_train, x_test_standard, y_test)
    mulnb = MultinomialNB()
    models_data['MultinomialNB'] = model_fit(mulnb, x_train_minmax, y_train, x_test_minmax, y_test)
    bnb = BernoulliNB()
    models_data['BernoulliNB'] = model_fit(bnb, x_train_standard, y_train, x_test_standard, y_test)
    conb = ComplementNB()
    models_data['ComplementNB'] = model_fit(conb, x_train_minmax, y_train, x_test_minmax, y_test)
    canb = CategoricalNB()
    models_data['CategoricalNB'] = model_fit(conb, x_train_minmax, y_train, x_test_minmax, y_test)

    # KNN
    # find the best outcome for different neighbors and standardization
    knn = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', p=1)
    models_data['Knn'] = model_fit(knn, x_train_standard, y_train, x_test_standard, y_test)

    # print the data
    for model in models_data:
        print(model, models_data[model])
'''
The results of these models

model           training dataset accuracy    test dataset accuracy
GaussianNB      [0.46151071929473053,        0.3897099832682655]
MultinomialNB   [0.5455019034261671,         0.46207473508087005]
BernoulliNB     [0.3759567220997796,         0.34453430005577246]
ComplementNB    [0.38280905630134243,        0.3846904629113218]
CategoricalNB   [0.38280905630134243,        0.3846904629113218]
Knn             [1.0,                        0.822643614054657]

'''