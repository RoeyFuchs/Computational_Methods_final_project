from sklearn import svm
from sklearn.model_selection import cross_validate
import numpy as np
from utils import plot_train_vald



def with_k_fold_with_regulariztion(train_data_x, train_data_y):
    prec = [0.2, 0.4, 0.6, 0.8, 1]
    acc_train = []
    acc_vald = []
    sampels_num = [int(x * len(train_data_x)) for x in prec]
    for k in sampels_num:
        clf = svm.SVC(C=10, kernel='poly', degree=3, gamma=1, coef0=0, max_iter=20000)  # c = 0 -> no penalety
        result = cross_validate(clf, train_data_x[:k], train_data_y[:k], cv=5, scoring='accuracy',
                                return_train_score=True)
        acc_vald.append(round(np.sum(result['test_score']) / len(result['test_score']), 3))
        acc_train.append(round(np.sum(result['train_score']) / len(result['train_score']), 3))
        print("vald: ", np.sum(result['test_score']) / len(result['test_score']))
        print("train: ", np.sum(result['train_score']) / len(result['train_score']))

    plot_train_vald(acc_train, acc_vald, sampels_num)

def witout_k_fold_with_regulariztion(train_data_x, train_data_y):
    training_set_size = int((4/5)*len(train_data_x))
    vald_x = train_data_x[training_set_size+1:]
    vald_y = train_data_y[training_set_size+1:]

    train_data_x = train_data_x[:training_set_size]
    train_data_y = train_data_y[:training_set_size]


    prec = [0.2, 0.4, 0.6, 0.8, 1]
    acc_train = []
    acc_vald = []
    sampels_num = [int(x * len(train_data_x)) for x in prec]
    for k in sampels_num:
        print(int(len(train_data_x) * k))
        clf = svm.SVC(C=10, kernel='poly', degree=3, gamma=1, coef0=0, max_iter=20000)

        clf.fit(train_data_x[:k], train_data_y[:k])
        y_hat = clf.predict(vald_x[:int(k/5)])
        acc_vald.append(np.sum(y_hat == vald_y[:int(k/5)]) / len(vald_y[:int(k/5)]))

        y_hat = clf.predict(train_data_x[:k])
        acc_train.append(np.sum(y_hat == train_data_y[:k]) / len(train_data_y[:k]))

    plot_train_vald(acc_train, acc_vald, sampels_num, cross=False)