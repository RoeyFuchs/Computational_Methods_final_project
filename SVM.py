from sklearn import svm
from sklearn.model_selection import cross_validate, GridSearchCV
import numpy as np
from sklearn.svm import SVC
from utils import plot_train_vald, get_reg_title

NO_REGULARIZATION = 150


def with_cross_validation(train_data_x, train_data_y, regularization=False):
    c = 1 if regularization else NO_REGULARIZATION
    percentage = [0.2, 0.4, 0.6, 0.8, 1]  # will learn with 20%, 40% ,.... 100% of the data
    acc_train = []
    acc_vald = []
    samples_num = [int(x * len(train_data_x) * (4 / 5)) for x in percentage]
    for k in samples_num:
        clf = clf = svm.SVC(C=c, kernel='poly', degree=1, gamma=1, coef0=0, max_iter=2000000)  # creat svm model
        result = cross_validate(clf, train_data_x[:k], train_data_y[:k], cv=5, scoring='accuracy',
                                return_train_score=True)  # using a part of the data every iteration.
        acc_vald.append(round(np.sum(result['test_score']) / len(result['test_score']), 3))
        acc_train.append(round(np.sum(result['train_score']) / len(result['train_score']), 3))
    plot_train_vald(acc_train, acc_vald, samples_num, x_label="Training set size (samples)",
                    y_label="Mean Accuracy (%)",
                    title="Mean Accuracy as function of training set size, cv (5), C={0}".format(get_reg_title(c)))


def without_cross_validation(train_data_x, train_data_y, regularization=False):
    c = 1 if regularization else NO_REGULARIZATION
    training_set_size = int((4 / 5) * len(train_data_x))
    vald_x = train_data_x[training_set_size + 1:]  # separate to validation set
    vald_y = train_data_y[training_set_size + 1:]
    train_data_x = train_data_x[:training_set_size]
    train_data_y = train_data_y[:training_set_size]
    percentage = [0.2, 0.4, 0.6, 0.8, 1]  # will learn with 20%, 40% ,.... 100% of the data
    acc_train = []
    acc_vald = []
    sampels_num = [int(x * len(train_data_x)) for x in percentage]
    for k in sampels_num:
        clf = svm.SVC(C=c, kernel='poly', degree=1, gamma=1, coef0=0, max_iter=2000000)  # creat svm model

        clf.fit(train_data_x[:k], train_data_y[:k])  # train the model
        y_hat = clf.predict(vald_x[:int(k / 5)])  # predict of the validation set
        acc_vald.append(np.sum(y_hat == vald_y[:int(k / 5)]) / len(vald_y[:int(k / 5)]))
        y_hat = clf.predict(train_data_x[:k])  # predict of the training set
        acc_train.append(np.sum(y_hat == train_data_y[:k]) / len(train_data_y[:k]))
    plot_train_vald(acc_train, acc_vald, sampels_num, x_label="Training set size (samples)", y_label="Accuracy (%)",
                    title="Accuracy as function of training set size, single validation set, C={}".format(
                        get_reg_title(c)))


# find best svm hyper-parameters using grid search
def find_best_svm_model(train_data_x, train_data_y):
    param_grid = [
        {'degree': [1, 2, 3, 4, 5], 'coef0': [0, 1, -1], 'gamma': [1, 0.5, 2, 4], 'kernel': ['poly']}
    ]
    clf = GridSearchCV(SVC(C=1, max_iter=20000), param_grid, scoring='accuracy', cv=5)
    clf.fit(train_data_x, train_data_y)
    return clf
