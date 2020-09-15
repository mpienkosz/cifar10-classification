import numpy as np
from sklearn.cross_validation import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from utils.data import split_data, sample_dataset
from utils.configuration import X_TRAIN_FEATURES_FILE, Y_TRAIN_FILE

X_train_features = np.load('%s.npy' % X_TRAIN_FEATURES_FILE)
Y_train = np.load('%s.npy' % Y_TRAIN_FILE)

X_train_features, Y_train = sample_dataset(X_train_features, Y_train, 0.1)
split_fn = ShuffleSplit(n=len(X_train_features), n_iter=1, test_size=0.25, random_state=0)

linear_params = {'C': [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3., 10.],
              'loss': ['hinge', 'squared_hinge'],
              'fit_intercept': [True, False]}
param_search = GridSearchCV(LinearSVC(), linear_params, cv=split_fn, verbose=2)
param_search.fit(X_train_features, Y_train)

print('Best parameters found:', param_search.best_params_)
print('', param_search.best_score_)
