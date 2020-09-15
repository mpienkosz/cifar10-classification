import os

# Dimensionality of the input images to pre-trained network
INPUT_IMG_SIZE = 139

# Paths for files with extracted features
FEATURES_SAVE_DIR = 'resources/inception_%spxl_feats' % (INPUT_IMG_SIZE)
X_TRAIN_FEATURES_FILE = os.path.join(FEATURES_SAVE_DIR, 'x_train_features')
X_TEST_FEATURES_FILE = os.path.join(FEATURES_SAVE_DIR, 'x_test_features')
Y_TRAIN_FILE = os.path.join(FEATURES_SAVE_DIR, 'y_train')
Y_TEST_FILE = os.path.join(FEATURES_SAVE_DIR, 'y_test')