import pandas as pd  # for load dataset
import numpy as np
from utils import remove_example_id, split_x_y, remove_unnecessary_features, plot_data
from sklearn.model_selection import train_test_split

DATA_FILE = "data/glass.data"
TEST_SET_SIZE = 0.1
TRAINING_SET_SIZE = 1 - TEST_SET_SIZE

raw_data = pd.read_csv(DATA_FILE, sep=",", header=None).to_numpy() # load data and convert to numpy array
data = remove_example_id(raw_data) # dataset has id for every example
data_x, data_y = split_x_y(data) # split features and labels
data_x = remove_unnecessary_features(data_x)



train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(data_x, data_y, test_size= TEST_SET_SIZE,
                                                                        shuffle=True) # split data to traning and
# test set

plot_data(train_data_x, train_data_y)







