import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# classes names
def get_dict_classes():
    return {0: "Cammeo", 1: "Osmancik"}

# split the data to x and y (features and class)
def split_x_y(data):
    x = data[:, 0:-1]
    y = data[:, -1]
    return x.astype('float'), y.astype('int')


# shuffle data, using sklearn shuffle
def shuffel(data_x, data_y):
    return shuffle(data_x, data_y)

# plot train and validation accuracy
def plot_train_vald(train, vald, x_axis, title, x_label, y_label):
    fig, ax = plt.subplots()
    train = [t * 100 for t in train]  # change to %
    vald = [v * 100 for v in vald]
    ax.plot(x_axis, train, '-ok', color='r', label='Train')
    ax.plot(x_axis, vald, '-ok', color='b', label='Validation')
    plt.xticks(x_axis, x_axis)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    ax.legend()
    plt.show()


# plot data for question 3
def plot_data(x, y):
    classes = get_dict_classes()
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in np.unique(y):  # add by classes
        a = np.where(y == i)
        a = x[a, :].squeeze()
        ax.scatter(a[:, 0], a[:, 1], a[:, 2], label=classes[i])
    ax.set_xlabel("Perimeter (pixels)")
    ax.set_ylabel("Major Axis Length (pixels)")
    ax.set_zlabel("Convex Area (pixels)")
    ax.legend(fontsize=8, ncol=1)
    plt.title("Binary classification by 3 relevant parameters")
    plt.show()


# get regulation number for plot (infinity symbol)
def get_reg_title(c):
    if c == sys.float_info.max:
        return "$\infty$"
    else:
        return str(c)


# get numbers from every culumns in a specific row
def column(matrix, i):
    return [row[i] for row in matrix]

# question 3
def remove_unnecessary_features(data):
    necessary_features = [1, 2, 5]
    return np.delete(data, [i for i in range(data.shape[1]) if i not in necessary_features], axis=1)

# question 3
def basic_visualization(data_x, data_y, TEST_SET_SIZE):
    data_x = remove_unnecessary_features(data_x)
    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(data_x, data_y, test_size=TEST_SET_SIZE,
                                                                            shuffle=True)
    plot_data(train_data_x, train_data_y)
