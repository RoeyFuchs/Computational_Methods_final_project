from random import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle


def get_dict_classes():
    return {0:"Cammeo", 1:"Osmancik"}

def remove_example_id(data):
    return np.delete(data, 0, axis=1) # remove the first column

def split_x_y(data):
    x = data[:, 0:-1]
    y = data[:, -1]
    return x.astype('float'), y.astype('int')

def remove_unnecessary_features(data):
    necessary_features = [1, 2, 5]
    return np.delete(data, [i for i in range(data.shape[1]) if i not in necessary_features],axis=1)



def split_k(data_x, data_y, k):
    new_x = np.array_split(data_x, k)
    new_y = np.array_split(data_y, k)
    return new_x, new_y

def shuffel(data_x, data_y):
    return shuffle(data_x, data_y)

def plot_train_vald(train, vald, x_axis, cross=True):
    fig, ax = plt.subplots()
    train = [t*100 for t in train] # change to %
    vald = [v*100 for v in vald]
    ax.plot(x_axis, train,'-ok', color='r', label='Train')
    ax.plot(x_axis, vald, '-ok', color='b', label='Validation')
    plt.xticks(x_axis, x_axis)
    plt.ylabel("Mean Accuracy (%)")
    plt.xlabel("Training set size (samples)")
    title = "Mean Accuracy as function of training set size"
    if cross:
        title += ", cross-validation (5)"
    plt.title(title)
    ax.legend()
    plt.show()


def plot_data(x, y):
    classes = get_dict_classes()
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in np.unique(y): # add by classes
        a = np.where(y == i)
        a = x[a, :].squeeze()
        ax.scatter(a[:,0], a[:, 1], a[:,2], label=classes[i])
    ax.set_xlabel("Perimeter (pixels)")
    ax.set_ylabel("Major Axis Length (pixels)")
    ax.set_zlabel("Convex Area (pixels)")
    ax.legend(fontsize=8, ncol=1)
    plt.title("Binary classification by 3 relevant parameters")
    plt.show()
