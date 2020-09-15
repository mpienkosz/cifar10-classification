import numpy as np
from keras.datasets import cifar10
from sklearn.svm import LinearSVC
from features.hog import HogFeatureExtractor

print('Loading cifar10..')
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

print('Extracting HoG features')
extractor = HogFeatureExtractor()
X_train_features = extractor.extract_features(X_train)
X_test_features = extractor.extract_features(X_test)

model = LinearSVC(verbose=1, C=3)

print('Training model...')
model.fit(X_train_features, Y_train)
print('Train score', model.score(X_train_features, Y_train))
print('Test score', model.score(X_test_features, Y_test))

