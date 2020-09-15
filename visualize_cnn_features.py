import numpy as np
from sklearn.manifold import TSNE
from utils.data import sample_dataset
from utils.visualization import plot_features
from utils.configuration import X_TRAIN_FEATURES_FILE, Y_TRAIN_FILE

VIZUALIZED_DATA_FRACTION = 0.1

print('Loading input feature data..')
X_train_features = np.load("%s.npy" % X_TRAIN_FEATURES_FILE)
Y_train = np.load("%s.npy" % Y_TRAIN_FILE)

#Sample 10% of original data
X_train_features, Y_train = sample_dataset(X_train_features, Y_train, VIZUALIZED_DATA_FRACTION)

print('Performing dimensionality reduction...')
model = TSNE(n_components=2, random_state=0, verbose=2)
X_proj= model.fit_transform(X_train_features)

print('Plotting..')
plot_features(X_proj, Y_train)

