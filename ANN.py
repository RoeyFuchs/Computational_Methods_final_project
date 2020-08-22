from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neural_network import MLPClassifier
import numpy as np
from utils import plot_train_vald

NO_REGULARIZATION = 0


def with_cross_validation(train_data_x, train_data_y, regularization=False):
    alpha = 0.015 if regularization else NO_REGULARIZATION
    percentage = [0.2, 0.4, 0.6, 0.8, 1] # will learn with 20%, 40% ,.... 100% of the data
    acc_train = []
    acc_vald = []
    sampels_num = [int(x * len(train_data_x) * (4 / 5)) for x in percentage]
    for k in sampels_num:
        # creat ann with one hidden layer (3 neurons)
        clf = MLPClassifier(hidden_layer_sizes=(3,), solver='sgd', activation='relu', alpha=alpha, max_iter=20000)
        result = cross_validate(clf, train_data_x[:k], train_data_y[:k], cv=5, scoring='accuracy',
                                return_train_score=True) # 5-cross-validation
        acc_vald.append(round(np.sum(result['test_score']) / len(result['test_score']), 3))
        acc_train.append(round(np.sum(result['train_score']) / len(result['train_score']), 3))
    plot_train_vald(acc_train, acc_vald, sampels_num, x_label="Training set size (samples)",
                    y_label="Mean Accuracy (%)",
                    title="Mean Accuracy as function of training set size, cv (5), $\\alpha$={0}".format(alpha))


def without_cross_validation(train_data_x, train_data_y, regularization=False):
    alpha = 0.015 if regularization else NO_REGULARIZATION
    training_set_size = int((4 / 5) * len(train_data_x))
    vald_x = train_data_x[training_set_size + 1:] # separate to validation set
    vald_y = train_data_y[training_set_size + 1:]

    train_data_x = train_data_x[:training_set_size]
    train_data_y = train_data_y[:training_set_size]

    percentage = [0.2, 0.4, 0.6, 0.8, 1] # will learn with 20%, 40% ,.... 100% of the data
    acc_train = []
    acc_vald = []
    sampels_num = [int(x * len(train_data_x)) for x in percentage]
    for k in sampels_num:
        clf = MLPClassifier(hidden_layer_sizes=(3,), solver='sgd', activation='relu', alpha=alpha, max_iter=20000)
        clf.fit(train_data_x[:k], train_data_y[:k]) # trainin g the model
        y_hat = clf.predict(vald_x[:int(k / 5)])
        acc_vald.append(np.sum(y_hat == vald_y[:int(k / 5)]) / len(vald_y[:int(k / 5)]))
        y_hat = clf.predict(train_data_x[:k]) # predict on the validation set
        acc_train.append(np.sum(y_hat == train_data_y[:k]) / len(train_data_y[:k]))
    plot_train_vald(acc_train, acc_vald, sampels_num, x_label="Training set size (samples)", y_label="Accuracy (%)",
                    title="Accuracy as function of training set size, single validation set, $\\alpha$={0}".format(
                        alpha))

# find best ann hyper-parameters using grid search
def find_best_ann_model(train_data_x, train_data_y):
    param_grid = [
        {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'learning_rate_init': [0.001, 0.01, 0.0001],
         'momentum': [
             0.9, 0.8, 0.99, 0.5], 'solver': ['sgd']}
    ]
    clf = GridSearchCV(MLPClassifier(hidden_layer_sizes=(3,),max_iter=20000, alpha=0.015), param_grid,
                       scoring='accuracy', cv=5)
    clf.fit(train_data_x, train_data_y)
    return clf
