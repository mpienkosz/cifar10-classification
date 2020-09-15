import os
import numpy as np
from features.inception import InceptionFeatureExtractor
from keras.datasets import cifar10 as keras_cifar10
from utils.configuration import FEATURES_SAVE_DIR, X_TRAIN_FEATURES_FILE, X_TEST_FEATURES_FILE, \
    Y_TRAIN_FILE, Y_TEST_FILE

if not os.path.isdir(FEATURES_SAVE_DIR):
    os.mkdir(FEATURES_SAVE_DIR)

print('Loading cifar10..')
(X_train, Y_train), (X_test, Y_test) = keras_cifar10.load_data()
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

print('Extracting Inception-v3 features..')
extractor = InceptionFeatureExtractor()
X_train_features = extractor.extract_features(X_train)
X_test_features = extractor.extract_features(X_test)

print('Saving feature data..')
np.save(X_TRAIN_FEATURES_FILE, X_train_features)
np.save(X_TEST_FEATURES_FILE, X_test_features)
np.save(Y_TRAIN_FILE, Y_train)
np.save(Y_TEST_FILE, Y_test)
