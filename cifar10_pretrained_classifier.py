import numpy as np
from sklearn.svm import LinearSVC
from utils.configuration import X_TRAIN_FEATURES_FILE, X_TEST_FEATURES_FILE, Y_TRAIN_FILE, Y_TEST_FILE

print('Loading input feature data..')
X_train_features = np.load('%s.npy' % X_TRAIN_FEATURES_FILE)
X_test_features = np.load('%s.npy' % X_TEST_FEATURES_FILE)
Y_train = np.load('%s.npy' % Y_TRAIN_FILE)
Y_test = np.load('%s.npy' % Y_TEST_FILE)

shallow_layer = LinearSVC(verbose=1, C=0.0003)

print('Training model..')
shallow_layer.fit(X_train_features, Y_train)
print('Train score', shallow_layer.score(X_train_features, Y_train))
print('Test score', shallow_layer.score(X_test_features, Y_test))

