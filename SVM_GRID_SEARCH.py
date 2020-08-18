from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn import svm
from sklearn.svm import SVC


def find_best_svm_model(train_data_x, train_data_y):
    param_grid = [
        {'degree': [1, 2, 3, 4, 5], 'coef0': [0, 1, -1], 'gamma': [1, 0.5, 2, 4], 'kernel': ['poly']}
    ]

    clf = GridSearchCV(SVC(max_iter=20000), param_grid, scoring='accuracy', cv=5)
    clf.fit(train_data_x, train_data_y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print()
