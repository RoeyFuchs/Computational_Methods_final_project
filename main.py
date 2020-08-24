import pandas as pd  # for load dataset
import numpy as np
from scipy.io import arff  # convert from arff to csv
from utils import split_x_y
import LR
import SVM
import ANN
from Ensemble_Model import Ensemble_Model
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
import os  # to get working directory

DATA_FILE = os.getcwd() + "\data\data.arff"  # data file name
TEST_SET_SIZE = 0.1  # 10% of the data - test set
TRAINING_SET_SIZE = 1 - TEST_SET_SIZE

with open(DATA_FILE, encoding='utf-8') as f:  # load arff file (data)
    data = arff.loadarff(f)
raw_data = pd.DataFrame(data[0]).to_numpy()  # load data and convert to numpy array

data_x, data_y = split_x_y(raw_data)  # split features and labels

scaler = StandardScaler()  # normalize the data
scaler.fit(data_x)
data_x = scaler.transform(data_x)

train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(data_x, data_y, test_size=TEST_SET_SIZE,
                                                                        shuffle=True)  # split data to traning and test set

y_hat_svm_result = []
y_hat_lr_result = []
y_hat_ann_result = []
y_hat_ensemble_result = []
# train and predict
for i in range(100):
    # get the best models (by grid search)
    svm = SVM.find_best_svm_model(train_data_x, train_data_y)
    ann = ANN.find_best_ann_model(train_data_x, train_data_y)
    lr = LR.get_LR_model(train_data_x, train_data_y)
    ensemble_model = Ensemble_Model([svm, ann, lr])
    # predict with every model
    y_hat_svm = svm.predict(test_data_x)
    y_hat_ann = ann.predict(test_data_x)
    y_hat_lr = lr.predict(test_data_x)
    y_hat_ensemble = ensemble_model.predict(test_data_x)
    # check resulets
    y_hat_svm_result.append((np.sum((y_hat_svm == test_data_y)) / len(test_data_y)) * 100)
    y_hat_ann_result.append((np.sum((y_hat_ann == test_data_y)) / len(test_data_y)) * 100)
    y_hat_lr_result.append((np.sum((y_hat_lr == test_data_y)) / len(test_data_y)) * 100)
    y_hat_ensemble_result.append((np.sum((y_hat_ensemble == test_data_y)) / len(test_data_y)) * 100)

print("svm: ", np.average(y_hat_svm_result))
print("lr: ", np.average(y_hat_lr_result))
print("ann: ", np.average(y_hat_ann_result))
print("ensemble: ", np.average(y_hat_ensemble_result))
