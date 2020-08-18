import pandas as pd  # for load dataset
import numpy as np
from scipy.io import arff  # convert from artff to csv
from utils import remove_example_id, split_x_y, remove_unnecessary_features, split_k, shuffel, plot_data, plot_train_vald
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from Q3 import q3
import os  # to get working directory
import torch  # ANN
from ANN import ANN
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
from SVM_NO_REGULARIZTION import *
from SVM_WITH_REGULARIZTION import *
from SVM_GRID_SEARCH import *

DATA_FILE = os.getcwd() + "\data\data.arff"
TEST_SET_SIZE = 0.1
TRAINING_SET_SIZE = 1 - TEST_SET_SIZE

with open(DATA_FILE, encoding='utf-8') as f:
    data = arff.loadarff(f)  # load arff file (data)
raw_data = pd.DataFrame(data[0]).to_numpy()  # load data and convert to numpy array

data_x, data_y = split_x_y(raw_data)  # split features and labels


'''# q3
q3(data_x, data_y, TEST_SET_SIZE)'''


scaler = StandardScaler()
scaler.fit(data_x)
data_x = scaler.transform(data_x)

'''# q3
q3(data_x, data_y, TEST_SET_SIZE)'''

train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(data_x, data_y, test_size=TEST_SET_SIZE,
                                                                        shuffle=True)  # split data to traning and
# test set


find_best_svm_model(train_data_x,train_data_y)
witout_k_fold_with_regulariztion(train_data_x, train_data_y)
witout_k_fold(train_data_x, train_data_y)


prec = [0.2, 0.4, 0.6, 0.8, 1]
acc_train = []
acc_vald = []
sampels_num = [int(x * len(train_data_x)) for x in prec]
for k in sampels_num:

    # SVM
    # degree, gamma = ha'kafol (gamma * x_i * x_j), r = (b independedt), c = regulariztion
    # train_x_cross, train_y_cross = split_k(train_data_x[:int(len(train_data_x) * k)], train_data_y[:int(len(
    # train_data_y)*k)], 5)
    print(int(len(train_data_x) * k))
    #for i in range(5):
    '''## k fold
    x = []
    y = []
    [x.append(train_x_cross[j].copy()) for j in range(5) if j != i]
    [y.append(train_y_cross[j].copy()) for j in range(5) if j != i]
    x_train = np.concatenate(np.array(x))
    y_train = np.concatenate(np.array(y))

    x_vald = np.array(train_x_cross[i])
    y_vald = np.array(train_y_cross[i])'''

    clf = svm.SVC(C=1, kernel='poly', degree=3,gamma=1,coef0=0, max_iter=20000)  # c = 0 -> no penalety

    # a = clf.fit(x_train, y_train)
    result = cross_validate(clf, train_data_x[:k], train_data_y[:k], cv=5, scoring='accuracy', return_train_score=True)
    '''y_hat = clf.predict(x_vald)
    print(np.sum(y_hat == y_vald) / len(y_vald))'''
    # result = cross_validate(clf, train_data_x, train_data_y, cv=5, scoring='accuracy', return_train_score=True)
    acc_vald.append(round(np.sum(result['test_score']) / len(result['test_score']), 3))
    acc_train.append(round(np.sum(result['train_score']) / len(result['train_score']),3))
    print("vald: ", np.sum(result['test_score']) / len(result['test_score']))
    print("train: ", np.sum(result['train_score']) / len(result['train_score']))

plot_train_vald(acc_train, acc_vald, sampels_num)


# ANN
# activtion function is a hyper parameter????????
# sgd: learning rate, momentum
# adam: learning rate, beta1, beta2, epsilon

train_x_cross, train_y_cross = split_k(train_data_x, train_data_y, 5)
models = []
for i in range(5):
    print("K: ", i)
    model = ANN()
    models.append(model)

    ## k fold
    x = []
    y = []
    [x.append(train_x_cross[j].copy()) for j in range(5) if j != i]
    [y.append(train_y_cross[j].copy()) for j in range(5) if j != i]
    x_train = np.concatenate(np.array(x))
    y_train = np.concatenate(np.array(y))

    x_vald = np.array(train_x_cross[i])
    y_vald = np.array(train_y_cross[i])

    learning_rate = 0.0001
    epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for ep in range(epochs):
        model.train()
        print("-----------", ep)
        x_train, y_train = shuffel(x_train, y_train)
        for example_x, example_y in zip(x_train, y_train):
            optimizer.zero_grad()
            output = model(torch.from_numpy(example_x).float())
            loss = F.nll_loss(output, torch.LongTensor([example_y]), reduction='sum')
            loss.backward()
            optimizer.step()

        sum = 0
        loss_sum = 0
        model.eval()
        for example_x, example_y in zip(x_vald, y_vald):
            output = model(torch.from_numpy(example_x).float())
            loss = F.nll_loss(output, torch.LongTensor([example_y]), reduction='sum')
            loss_sum += loss.item()
            if output.max(1, keepdim=True)[1] == example_y:
                sum = sum + 1
        print(sum / len(y_vald))
        print(loss_sum / len(y_vald))
