import pandas as pd  # for load dataset
import numpy as np
from scipy.io import arff #convert from artff to csv
from utils import remove_example_id, split_x_y, remove_unnecessary_features,split_k,shuffel, plot_data
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import svm
from Q3 import q3
import os # to get working directory
import torch # ANN
from ANN import ANN
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch


DATA_FILE = os.getcwd()+"\data\data.arff"
TEST_SET_SIZE = 0.1
TRAINING_SET_SIZE = 1 - TEST_SET_SIZE

with open(DATA_FILE, encoding='utf-8') as f:
    data = arff.loadarff(f) # load arff file (data)
raw_data = pd.DataFrame(data[0]).to_numpy() # load data and convert to numpy array

data_x, data_y = split_x_y(raw_data) # split features and labels

'''
# q3
q3(data_x, data_y, TEST_SET_SIZE)
'''

train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(data_x, data_y, test_size= TEST_SET_SIZE,
                                                                        shuffle=True) # split data to traning and
# test set


# SVM
'''for i in range(5):
    clf = svm.SVC(C=0.1, kernel='rbf',gamma=0.1, max_iter=5000) # c = 0 -> no penalety
    # clf.fit(train_data_x, train_data_y)
    result = cross_validate(clf, train_data_x, train_data_y, cv=5)
    print(np.sum(result['test_score'])/len(result['test_score']))'''


# ANN

train_x_cross, train_y_cross = split_k(train_data_x, train_data_y, 5)
models = []
for i in range(5):
    print("K: ",i)
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

    model.train()
    for ep in range(epochs):
        print("-----------", ep)
        x_train, y_train = shuffel(x_train, y_train)
        for example_x, example_y in zip(x_train, y_train):
            optimizer.zero_grad()
            output = model(torch.from_numpy(example_x).float())
            loss = F.nll_loss(output,  torch.LongTensor([example_y]), reduction='sum')
            loss.backward()
            optimizer.step()

        sum = 0
        for example_x, example_y in zip(x_vald, y_vald):
            output = model(torch.from_numpy(example_x).float())
            if output.max(1, keepdim=True)[1] == example_y:
                sum = sum + 1
        print(sum/len(y_vald))



'''
sum = 0
model.eval()
sumZ=0
sumO = 0


    if output.max(1, keepdim=True)[1] == 0:
        sumZ = sumZ+1
    else:
        sumO = sumO +1


print(sumO)
print(sumZ)
print(sum/len(test_data_y))'''

