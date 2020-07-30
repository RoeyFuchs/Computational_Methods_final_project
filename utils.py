import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_dict_classes():
    return {1:"building_windows_float_processed", 2:"building_windows_non_float_processed",
            3:"vehicle_windows_float_processed", 4:"vehicle_windows_non_float_processed", 5:"containers",
            6:"tableware", 7:"headlamps"}

def remove_example_id(data):
    return np.delete(data, 0, axis=1) # remove the first column

def split_x_y(data):
    x = data[:, 0:-1]
    y = data[:, -1]
    return x, y

def remove_unnecessary_features(data):
    necessary_features = [2, 3, 7]
    return np.delete(data, [i for i in range(data.shape[1]) if i not in necessary_features],axis=1)


def plot_data(x, y):
    classes = get_dict_classes()
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in np.unique(y): # add by classes
        a = np.where(y == i)
        a = x[a, :].squeeze()
        ax.scatter(a[:,0], a[:, 1], a[:,2], label=classes[i])
    ax.set_xlabel("Mg")
    ax.set_ylabel("Al")
    ax.set_zlabel("Ba")
    ax.legend(fontsize=8, ncol=2)
    plt.show()
