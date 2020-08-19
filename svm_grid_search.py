from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn import svm
from sklearn.svm import SVC


def find_best_svm_model(train_data_x, train_data_y):
    param_grid = [
        {'degree': [1, 2, 3, 4, 5], 'coef0': [0, 1, -1], 'gamma': [1, 0.5, 2, 4], 'kernel': ['poly']}
    ]
    clf = GridSearchCV(SVC(max_iter=20000), param_grid, scoring='accuracy', cv=5)
    clf.fit(train_data_x, train_data_y)
    return clf
