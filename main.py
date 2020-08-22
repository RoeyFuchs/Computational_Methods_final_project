import pandas as pd  # for load dataset
import numpy as np
from scipy.io import arff  # convert from artff to csv

import LR
import SVM
import ANN
from utils import remove_example_id, split_x_y, remove_unnecessary_features, split_k, shuffel, plot_data, plot_train_vald
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from Q3 import q3
import os  # to get working directory
import torch  # ANN
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch

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


train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(data_x, data_y, test_size=TEST_SET_SIZE,
                                                                        shuffle=True)  # split data to traning and test set

svm = SVM.find_best_svm_model(train_data_x, train_data_y)
ann = ANN.find_best_ann_model(train_data_x,train_data_y)
lr = LR.get_LR_model(train_data_x, train_data_y)

y_hat_svm = svm.predict(test_data_x)
y_hat_ann = ann.predict(test_data_x)
y_hat_lr = lr.predict(test_data_x)

print("svm: ", np.sum(y_hat_svm == test_data_y)*100)
print("ann: ", np.sum(y_hat_ann == test_data_y)*100)
print("lr: ", np.sum(y_hat_lr == test_data_y)*100)