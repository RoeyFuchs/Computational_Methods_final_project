from sklearn.model_selection import train_test_split

from utils import remove_example_id, split_x_y, remove_unnecessary_features, plot_data

def q3(data_x, data_y, TEST_SET_SIZE):
    data_x = remove_unnecessary_features(data_x)
    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(data_x, data_y, test_size=TEST_SET_SIZE,
                                                                            shuffle=True)  # split data to traning and
    plot_data(train_data_x, train_data_y)
