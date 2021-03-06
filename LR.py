from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import numpy as np
from utils import plot_train_vald, get_reg_title
import sys

NO_REGULARIZATION = sys.float_info.max  # the biggest number that can fit to float


def with_cross_validation(train_data_x, train_data_y, regularization=False):
    c = 0.0005 # if regularization else NO_REGULARIZATION
    pen = 'l2' if regularization else 'none'
    percentage = [0.2, 0.4, 0.6, 0.8, 1]  # will learn with 20%, 40% ,.... 100% of the data
    acc_train = []
    acc_vald = []
    sampels_num = [int(x * len(train_data_x) * (4 / 5)) for x in percentage]
    for k in sampels_num:
        clf = LogisticRegression(C=c, max_iter=20000, penalty=pen)  # creat LR model
        result = cross_validate(clf, train_data_x[:k], train_data_y[:k], cv=5, scoring='accuracy',
                                return_train_score=True)  # 5-cross-validation
        acc_vald.append(round(np.sum(result['test_score']) / len(result['test_score']), 3))
        acc_train.append(round(np.sum(result['train_score']) / len(result['train_score']), 3))
    plot_train_vald(acc_train, acc_vald, sampels_num, x_label="Training set size (samples)",
                    y_label="Mean Accuracy (%)",
                    title="Mean Accuracy as function of training set size, cv (5), C={0}".format(get_reg_title(c)))


def without_cross_validation(train_data_x, train_data_y, regularization=False):
    c = 10 # if regularization else NO_REGULARIZATION
    pen = 'l2' if regularization else 'none'
    training_set_size = int((4 / 5) * len(train_data_x))
    vald_x = train_data_x[training_set_size + 1:]
    vald_y = train_data_y[training_set_size + 1:]
    train_data_x = train_data_x[:training_set_size]  # separate to validation set
    train_data_y = train_data_y[:training_set_size]
    percentage = [0.2, 0.4, 0.6, 0.8, 1]  # will learn with 20%, 40% ,.... 100% of the data
    acc_train = []
    acc_vald = []
    sampels_num = [int(x * len(train_data_x)) for x in percentage]
    for k in sampels_num:
        print(int(len(train_data_x) * k))
        clf = LogisticRegression(C=c, max_iter=20000, penalty=pen)
        clf.fit(train_data_x[:k], train_data_y[:k])  # learn the data
        y_hat = clf.predict(vald_x[:int(k / 5)])  # predict on the validation set
        acc_vald.append(np.sum(y_hat == vald_y[:int(k / 5)]) / len(vald_y[:int(k / 5)]))
        y_hat = clf.predict(train_data_x[:k])
        acc_train.append(np.sum(y_hat == train_data_y[:k]) / len(train_data_y[:k]))
    plot_train_vald(acc_train, acc_vald, sampels_num, x_label="Training set size (samples)", y_label="Accuracy (%)",
                    title="Accuracy as function of training set size, single validation set, C={}".format(
                        get_reg_title(c)))


def get_LR_model(train_data_x, train_data_y):
    clf = LogisticRegression(C=2, max_iter=20000)
    clf.fit(train_data_x, train_data_y)
    return clf  # get the LR model after training on data
